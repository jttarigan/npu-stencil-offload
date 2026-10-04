// InkBench for iOS: the netime / metaltime protocol on a phone, so every device
// gets the same per-submission cost model as the Mac (inklab/netime/sweep.sh).
//
// Phase A, every workload x K in {1,2,4,8}:
//   Core ML on ne / gpu / cpu, back to back, 3 x 300 submissions
//   Metal (K steps per command buffer), back to back, 3 x 300: wall + GPU execution
//   Core ML on ne, paced at the game cadence (one submission every K/30 s), 2 x 10 s
// Phase B, the pacing control (ink and heat_s32 at K = 1 and 8):
//   Core ML cpu / gpu and Metal, paced, 2 x 10 s
// Rows go to Documents/bench_<model>_<time>.csv (sweep.sh columns plus device and
// thermal state), and first Documents/placement_<model>_<time>.csv: where Core ML
// places each model's operations (MLComputePlan). Launch argument "quick" runs one
// repeat of everything (smoke test); "placement" writes the placement CSV and stops.
//
// Same caveat as the game: the file is main.swift, so no @main; the entry point is
// the UIApplicationMain call at the bottom.
import UIKit
import CoreML
import Metal

let quick = CommandLine.arguments.contains("quick")
let workloads: [(stem: String, kernel: String, sweeps: Int)] = [
    ("ink_unroll_128x256_j26_fp16", "ink", 0),
    ("heat_128x256_s32_fp16", "heat", 32),
    ("heat_128x256_s8_fp16", "heat", 8),
    ("grayscott_128x256_s8_fp16", "grayscott", 8),
]
let ks = [1, 2, 4, 8]

func deviceModel() -> String {
    var s = utsname(); uname(&s)
    return withUnsafeBytes(of: &s.machine) { String(cString: $0.bindMemory(to: CChar.self).baseAddress!) }
}
func thermal() -> Int { ProcessInfo.processInfo.thermalState.rawValue }   // 0 nominal .. 3 critical
func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

// MARK: - Core ML runner (as netime: preallocated float32 inputs, random small values)

final class CoreMLRunner {
    let model: MLModel
    let provider: MLDictionaryFeatureProvider
    init(url: URL, units: String) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units == "cpu" ? .cpuOnly : units == "gpu" ? .cpuAndGPU : .cpuAndNeuralEngine
        model = try MLModel(contentsOf: url, configuration: cfg)
        var inputs: [String: MLFeatureValue] = [:]
        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            let arr = try MLMultiArray(shape: desc.multiArrayConstraint!.shape, dataType: .float32)
            let p = arr.dataPointer.bindMemory(to: Float.self, capacity: arr.count)
            for i in 0..<arr.count { p[i] = Float.random(in: -1...1) * 0.1 }
            inputs[name] = MLFeatureValue(multiArray: arr)
        }
        provider = try MLDictionaryFeatureProvider(dictionary: inputs)
    }
    func submit() throws { _ = try model.prediction(from: provider) }
}

// MARK: - Metal runner (as metaltime: the game's ink kernels, K steps per command buffer)

final class MetalRunner {
    let W = 128, H = 256, jacobi = 26
    let kernel: String, sweeps: Int, K: Int
    let queue: MTLCommandQueue
    var pso: [String: MTLComputePipelineState] = [:]
    var vel: [MTLTexture], dye: [MTLTexture], pr: [MTLTexture], div: MTLTexture, fld: [MTLTexture]
    let splatBuf: MTLBuffer
    var u: (SIMD4<Float>, SIMD4<Float>)
    var gpuSec = 0.0

    init(dev: MTLDevice, lib: MTLLibrary, kernel: String, sweeps: Int, K: Int) throws {
        self.kernel = kernel; self.sweeps = sweeps; self.K = K
        queue = dev.makeCommandQueue()!
        for n in ["ink_step", "ink_vorticity", "ink_divergence", "ink_jacobi", "ink_project", "heat", "grayscott"] {
            pso[n] = try dev.makeComputePipelineState(function: lib.makeFunction(name: n)!)
        }
        let (w, h) = (128, 256)
        func tex(_ f: MTLPixelFormat) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
            return dev.makeTexture(descriptor: d)!
        }
        vel = [tex(.rg16Float), tex(.rg16Float)]; dye = [tex(.rgba16Float), tex(.rgba16Float)]
        pr = [tex(.r16Float), tex(.r16Float)]; div = tex(.r16Float)
        let ff: MTLPixelFormat = kernel == "heat" ? .r16Float : .rg16Float
        fld = [tex(ff), tex(ff)]
        // Two splats per step, the same load as metaltime and the Android GLES path.
        let splats: [Float] = [40, 60, 20, -10, 0.9, 0.4, 0.2, 0, 3, 0.9, 0, 0,
                               90, 180, 0, 0, 1, 1, 1, 0, 4, 0.6, 5, 0]
        splatBuf = dev.makeBuffer(bytes: splats, length: splats.count * 4)!
        u = (SIMD4(1 / 30, 0.3, 0.03, 2), SIMD4(2.5, Float(w), Float(h), 0))
    }

    func submit() {
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let grid = MTLSize(width: (W + 7) / 8, height: (H + 7) / 8, depth: 1)
        for _ in 0..<K {
            if kernel == "ink" {
                e.setComputePipelineState(pso["ink_step"]!)
                e.setTexture(vel[0], index: 0); e.setTexture(dye[0], index: 1)
                e.setTexture(vel[1], index: 2); e.setTexture(dye[1], index: 3)
                e.setBytes(&u, length: 32, index: 0); e.setBuffer(splatBuf, offset: 0, index: 1)
                e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                e.setComputePipelineState(pso["ink_vorticity"]!)
                e.setTexture(vel[1], index: 0); e.setTexture(vel[0], index: 1)
                e.setBytes(&u, length: 32, index: 0); e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                e.setComputePipelineState(pso["ink_divergence"]!)
                e.setTexture(vel[0], index: 0); e.setTexture(div, index: 1)
                e.setBytes(&u, length: 32, index: 0); e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                e.setComputePipelineState(pso["ink_jacobi"]!)
                e.setBytes(&u, length: 32, index: 0)
                for _ in 0..<jacobi {
                    e.setTexture(pr[0], index: 0); e.setTexture(div, index: 1); e.setTexture(pr[1], index: 2)
                    e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg); pr.swapAt(0, 1)
                }
                e.setComputePipelineState(pso["ink_project"]!)
                e.setTexture(vel[0], index: 0); e.setTexture(pr[0], index: 1); e.setTexture(vel[1], index: 2)
                e.setBytes(&u, length: 32, index: 0); e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                vel.swapAt(0, 1); dye.swapAt(0, 1)
            } else {
                e.setComputePipelineState(pso[kernel]!)
                for _ in 0..<sweeps {
                    e.setTexture(fld[0], index: 0); e.setTexture(fld[1], index: 1)
                    e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg); fld.swapAt(0, 1)
                }
            }
        }
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        gpuSec += cb.gpuEndTime - cb.gpuStartTime
    }
}

// MARK: - protocol

final class Bench {
    let log: (String) -> Void
    let out: URL
    let dev = MTLCreateSystemDefaultDevice()!
    let lib: MTLLibrary
    let reps: Int, pacedReps: Int, calls: Int, pacedSeconds: Double

    init(log: @escaping (String) -> Void) throws {
        self.log = log
        reps = quick ? 1 : 3; pacedReps = quick ? 1 : 2; calls = quick ? 50 : 300; pacedSeconds = quick ? 3 : 10
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        out = docs.appendingPathComponent("bench_\(deviceModel())_\(Int(Date().timeIntervalSince1970))\(quick ? "_quick" : "").csv")
        try "device,workload,K,units,mode,rep,ms_per_submission,thermal\n".write(to: out, atomically: true, encoding: .utf8)
        lib = try dev.makeLibrary(source: benchKernelSource, options: nil)
    }

    func row(_ w: String, _ k: Int, _ units: String, _ mode: String, _ rep: Int, _ ms: Double) {
        let line = "\"\(deviceModel())\",\(w),\(k),\(units),\(mode),\(rep),\(String(format: "%.4f", ms)),\(thermal())\n"
        if let h = try? FileHandle(forWritingTo: out) { h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); try? h.close() }
    }

    /// Back to back: ms per submission (wall).
    func b2b(_ submit: () throws -> Void) rethrows -> Double {
        let t0 = now()
        for _ in 0..<calls { try submit() }
        return Double(now() - t0) / 1e6 / Double(calls)
    }

    /// Paced at 30/K Hz: busy ms per submission.
    func paced(_ k: Int, _ submit: () throws -> Void) rethrows -> Double {
        let n = max(Int(pacedSeconds * 30 / Double(k)), 5)
        let period = UInt64(Double(k) / 30 * 1e9)
        var busy: UInt64 = 0
        let t0 = now()
        for i in 0..<n {
            let a = now(); try submit(); busy += now() - a
            let next = t0 + UInt64(i + 1) * period
            let t = now(); if next > t { usleep(UInt32((next - t) / 1000)) }
        }
        return Double(busy) / 1e6 / Double(n)
    }

    func url(_ stem: String, _ k: Int) -> URL {
        Bundle.main.url(forResource: "\(stem)_u\(k)", withExtension: "mlmodelc", subdirectory: "BenchModels")!
    }

    /// Where Core ML places each model's operations under cpuAndNeuralEngine
    /// (MLComputePlan), written next to the timing CSV before the sweep.
    func placement() {
        guard #available(iOS 17.4, *) else { log("placement: needs iOS 17.4"); return }
        let pout = out.deletingLastPathComponent().appendingPathComponent(
            out.lastPathComponent.replacingOccurrences(of: "bench_", with: "placement_"))
        let sem = DispatchSemaphore(value: 0)
        var text = "device," + placementHeader + "\n"
        Task {
            for w in workloads { for k in ks {
                do { text += "\"\(deviceModel())\"," + (try await placementRow(url(w.stem, k), units: .cpuAndNeuralEngine)) + "\n" }
                catch { text += "\"\(deviceModel())\",\(w.stem)_u\(k),ERROR \(error)\n" }
            } }
            sem.signal()
        }
        sem.wait()
        try? text.write(to: pout, atomically: true, encoding: .utf8)
        let offNE = text.split(separator: "\n").dropFirst().filter { !$0.contains(",1.000,0.000,0.000,") }.count
        log("placement: \(offNE) of \(workloads.count * ks.count) models not fully on the neural engine")
    }

    func run() {
        log("InkBench on \(deviceModel()), iOS \(UIDevice.current.systemVersion)\(quick ? " (quick)" : "")")
        placement()
        if CommandLine.arguments.contains("placement") { log("DONE placement only"); return }
        // Phase A
        for w in workloads { for k in ks {
            for units in ["ne", "gpu", "cpu"] {
                do {
                    let r = try CoreMLRunner(url: url(w.stem, k), units: units)
                    for _ in 0..<20 { try r.submit() }
                    for rep in 1...reps { row(w.stem, k, units, "b2b", rep, try b2b { try r.submit() }) }
                    if units == "ne" {
                        for rep in 1...pacedReps { row(w.stem, k, "ne", "paced", rep, try paced(k) { try r.submit() }) }
                    }
                } catch { log("FAIL \(w.stem) K=\(k) \(units): \(error)") }
            }
            do {
                let m = try MetalRunner(dev: dev, lib: lib, kernel: w.kernel, sweeps: w.sweeps, K: k)
                for _ in 0..<20 { m.submit() }
                for rep in 1...reps {
                    m.gpuSec = 0
                    let ms = b2b { m.submit() }
                    row(w.stem, k, "metal", "b2b", rep, ms)
                    row(w.stem, k, "metal-exec", "b2b", rep, m.gpuSec * 1e3 / Double(calls))
                }
            } catch { log("FAIL \(w.stem) K=\(k) metal: \(error)") }
            log("A \(w.stem.split(separator: "_").first!) K=\(k) done, thermal \(thermal())")
        } }
        // Phase B: the pacing control
        for w in workloads where w.stem.hasPrefix("ink") || w.stem.hasPrefix("heat_128x256_s32") {
            for k in [1, 8] {
                for units in ["cpu", "gpu"] {
                    do {
                        let r = try CoreMLRunner(url: url(w.stem, k), units: units)
                        for _ in 0..<10 { try r.submit() }
                        for rep in 1...pacedReps { row(w.stem, k, units, "paced", rep, try paced(k) { try r.submit() }) }
                    } catch { log("FAIL paced \(w.stem) K=\(k) \(units): \(error)") }
                }
                do {
                    let m = try MetalRunner(dev: dev, lib: lib, kernel: w.kernel, sweeps: w.sweeps, K: k)
                    for _ in 0..<10 { m.submit() }
                    for rep in 1...pacedReps {
                        m.gpuSec = 0
                        let n = max(Int(pacedSeconds * 30 / Double(k)), 5)
                        let ms = paced(k) { m.submit() }
                        row(w.stem, k, "metal", "paced", rep, ms)
                        row(w.stem, k, "metal-exec", "paced", rep, m.gpuSec * 1e3 / Double(n))
                    }
                } catch { log("FAIL paced \(w.stem) K=\(k) metal: \(error)") }
                log("B \(w.stem.split(separator: "_").first!) K=\(k) done, thermal \(thermal())")
            }
        }
        log("DONE \(out.lastPathComponent)")
    }
}

// MARK: - app shell

final class ViewController: UIViewController {
    let label = UILabel()
    var lines: [String] = []
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        label.textColor = .white; label.numberOfLines = 0
        label.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        label.frame = view.bounds.insetBy(dx: 16, dy: 60)
        label.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(label)
        UIApplication.shared.isIdleTimerDisabled = true
        Thread {
            do { try Bench(log: self.show).run() } catch { self.show("FAIL setup: \(error)") }
        }.start()
    }
    func show(_ s: String) {
        NSLog("InkBench: %@", s)
        DispatchQueue.main.async {
            self.lines.append(s); if self.lines.count > 40 { self.lines.removeFirst() }
            self.label.text = self.lines.joined(separator: "\n")
        }
    }
}

// Scene lifecycle: iOS 27 terminates apps that have not adopted UIScene, so the
// window is created per scene. The delegate class is supplied in code, which
// keeps Info.plist free of the module-qualified class name.
final class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     configurationForConnecting session: UISceneSession,
                     options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        let c = UISceneConfiguration(name: nil, sessionRole: session.role)
        c.delegateClass = SceneDelegate.self
        return c
    }
}

final class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?
    func scene(_ scene: UIScene, willConnectTo session: UISceneSession,
               options connectionOptions: UIScene.ConnectionOptions) {
        guard let ws = scene as? UIWindowScene else { return }
        let w = UIWindow(windowScene: ws)
        w.rootViewController = ViewController()
        w.makeKeyAndVisible()
        window = w
    }
}

UIApplicationMain(CommandLine.argc, CommandLine.unsafeArgv, nil, NSStringFromClass(AppDelegate.self))
