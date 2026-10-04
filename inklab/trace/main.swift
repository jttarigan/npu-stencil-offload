// Golden fluid trace: runs the game's ink kernels (../ink_kernels.metal,
// wrapped by run.sh) headlessly on a seeded, scripted splat
// sequence at the frozen reference configuration, and dumps the fields.
//
//   trace_harness <outDir> [steps=300] [dumpEvery=30] [seed=12345]
//
// Output: <outDir>/manifest.json plus vel_NNNN.f32 (W*H*2), dye_NNNN.f32
// (W*H*3), pr_NNNN.f32 (W*H) row-major (y, x), float32 little-endian.
// Also writes gpu_ms.csv: per-step GPU time of the whole fluid step.
import Foundation
import Metal

// ---- frozen reference configuration (mirrors the game's renderer) ----
let W = 128, H = 256
let dt: Float = 1.0 / 30.0
let velDamp: Float = 0.3, dyeDamp: Float = 0.03, vorticity: Float = 2.5
let jacobiIters = 26
let maxSplats = 64

struct InkSplatGPU { var posVel: SIMD4<Float>; var color: SIMD4<Float>; var params: SIMD4<Float> }
struct InkUniforms { var a: SIMD4<Float>; var b: SIMD4<Float> }

// ---- deterministic splat script (SplitMix64, same generator family as the game) ----
struct SplitMix { var s: UInt64
    mutating func next() -> UInt64 { s &+= 0x9E3779B97F4A7C15; var z = s
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) &* 0x94D049BB133111EB; return z ^ (z >> 31) }
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }   // [0,1) with 24 bits, power-of-two scaling
}
let hues: [Float] = [0.161, 0.507, 0.427, 0.786, 0.920]   // the five pellet hues
func hsv(_ h: Float) -> SIMD3<Float> {   // s = 1, v = 1
    let i = Int(h * 6) % 6, f = h * 6 - Float(Int(h * 6))
    let q = 1 - f, t = f
    switch i { case 0: return [1, t, 0]; case 1: return [q, 1, 0]; case 2: return [0, 1, t]
               case 3: return [0, q, 1]; case 4: return [t, 0, 1]; default: return [1, 0, q] }
}
/// Splats for step n: an eat-like drop every 12 steps, a head stir every step
/// (orbiting), a finale-like radial burst at step 150. Values in sim texels.
func splats(step n: Int, rng: inout SplitMix) -> [InkSplatGPU] {
    var out: [InkSplatGPU] = []
    let t = Float(n) * dt
    let cx = Float(W) * (0.5 + 0.35 * cos(t * 0.9)), cy = Float(H) * (0.5 + 0.35 * sin(t * 0.7))
    // head stir: velocity only (the game's per-frame stir under the player's head)
    out.append(InkSplatGPU(posVel: [cx, cy, -18 * sin(t * 0.9), 15 * cos(t * 0.7)], color: .zero,
                           params: [3.0, 0, 0, 0]))
    if n % 12 == 0 {   // pellet eaten: dye drop with impulse along travel
        let h = hues[Int(rng.next() % 5)], c = hsv(h)
        let px = 8 + rng.unit() * Float(W - 16), py = 8 + rng.unit() * Float(H - 16)
        let ang = rng.unit() * 6.2831853
        out.append(InkSplatGPU(posVel: [px, py, 12 * cos(ang), 12 * sin(ang)], color: [c.x, c.y, c.z, 0],
                               params: [0.9 * 11.6, 0.9, 0, 0]))          // radius 0.9 cells ~ 11.6 texels/cell
    }
    if n == 150 {      // finale-like burst: 12 radial splats
        for _ in 0..<12 {
            let h = hues[Int(rng.next() % 5)], c = hsv(h)
            let px = 8 + rng.unit() * Float(W - 16), py = 8 + rng.unit() * Float(H - 16)
            out.append(InkSplatGPU(posVel: [px, py, 0, 0], color: [c.x, c.y, c.z, 0],
                                   params: [1.2 * 11.6, 0.85, 4 * 11.6, 0]))
        }
    }
    return Array(out.prefix(maxSplats))
}

// ---- Metal setup ----
let args = CommandLine.arguments
let outDir = args.count > 1 ? args[1] : "trace_out"
let dumpEvery = args.count > 3 ? Int(args[3])! : 30
let seed = args.count > 4 ? UInt64(args[4])! : 12345
// Optional 5th argument: a recorded gameplay script (record_gameplay.swift)
// replaces the synthetic script; its step count overrides `steps`.
var recorded: [[InkSplatGPU]]? = nil
if args.count > 5 {
    let j = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[5]))) as! [String: Any]
    recorded = (j["script"] as! [[String: Any]]).map { e in
        (e["splats"] as! [[String: Any]]).map { s in
            let pv = (s["posVel"] as! [Double]).map(Float.init), c = (s["color"] as! [Double]).map(Float.init), pa = (s["params"] as! [Double]).map(Float.init)
            return InkSplatGPU(posVel: SIMD4(pv[0], pv[1], pv[2], pv[3]), color: SIMD4(c[0], c[1], c[2], 0), params: SIMD4(pa[0], pa[1], pa[2], 0)) } }
}
let steps = recorded?.count ?? (args.count > 2 ? Int(args[2])! : 300)
// PACE_HZ env: run at a fixed cadence (sleep between steps) for energy runs
let paceHz = Double(ProcessInfo.processInfo.environment["PACE_HZ"] ?? "0") ?? 0
let paceStart = DispatchTime.now().uptimeNanoseconds
try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { fatalError("no Metal") }
let library = try device.makeLibrary(source: shaderSource, options: nil)
var pipelines: [String: MTLComputePipelineState] = [:]
for name in ["ink_step", "ink_vorticity", "ink_divergence", "ink_jacobi", "ink_project", "ink_zero"] {
    pipelines[name] = try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}
func tex(_ format: MTLPixelFormat) -> MTLTexture {
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: W, height: H, mipmapped: false)
    d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .shared
    return device.makeTexture(descriptor: d)!
}
// STORAGE=half reproduces the SHIPPED texture formats (rg16Float/rgba16Float/
// r16Float: fp32 arithmetic in the kernels, fp16 storage between passes).
let half = ProcessInfo.processInfo.environment["STORAGE"] == "half"
let vel = [tex(half ? .rg16Float : .rg32Float), tex(half ? .rg16Float : .rg32Float)]
let dye = [tex(half ? .rgba16Float : .rgba32Float), tex(half ? .rgba16Float : .rgba32Float)]
let pr = [tex(half ? .r16Float : .r32Float), tex(half ? .r16Float : .r32Float)], div = tex(half ? .r16Float : .r32Float)
var velCur = 0, dyeCur = 0, prCur = 0

func readTex(_ t: MTLTexture, channels: Int, stride: Int) -> [Float] {   // returns channels per texel, row-major
    var buf = [Float](repeating: 0, count: W * H * stride)
    if half {
        var h = [Float16](repeating: 0, count: W * H * stride)
        t.getBytes(&h, bytesPerRow: W * stride * 2, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0)
        for i in 0..<h.count { buf[i] = Float(h[i]) }
    } else {
        t.getBytes(&buf, bytesPerRow: W * stride * 4, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0)
    }
    if channels == stride { return buf }
    var out = [Float](); out.reserveCapacity(W * H * channels)
    for i in 0..<(W * H) { for c in 0..<channels { out.append(buf[i * stride + c]) } }
    return out
}
func dump(_ name: String, _ data: [Float]) {
    data.withUnsafeBufferPointer { p in
        try! Data(buffer: p).write(to: URL(fileURLWithPath: "\(outDir)/\(name)")) }
}

// clear
do { let cmd = queue.makeCommandBuffer()!, ce = cmd.makeComputeCommandEncoder()!
    for t in vel + dye + pr + [div] {
        ce.setComputePipelineState(pipelines["ink_zero"]!); ce.setTexture(t, index: 0)
        ce.dispatchThreadgroups(MTLSize(width: W / 8, height: H / 8, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    }
    ce.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
}

var rng = SplitMix(s: seed)
var gpuCsv = "step,splats,gpuMs\n"
var manifest: [[String: Any]] = []
let tg = MTLSize(width: 8, height: 8, depth: 1), groups = MTLSize(width: W / 8, height: H / 8, depth: 1)
for n in 0..<steps {
    let sp = recorded?[n] ?? splats(step: n, rng: &rng)
    var u = InkUniforms(a: [dt, velDamp, dyeDamp, Float(sp.count)], b: [vorticity, Float(W), Float(H), 0])
    let cmd = queue.makeCommandBuffer()!, ce = cmd.makeComputeCommandEncoder()!
    func run(_ name: String, _ ts: [MTLTexture], splats s: [InkSplatGPU]? = nil) {
        ce.setComputePipelineState(pipelines[name]!)
        for (i, t) in ts.enumerated() { ce.setTexture(t, index: i) }
        ce.setBytes(&u, length: MemoryLayout<InkUniforms>.stride, index: 0)
        if let s = s { var arr = s; if arr.isEmpty { arr = [InkSplatGPU(posVel: .zero, color: .zero, params: .zero)] }
            ce.setBytes(arr, length: MemoryLayout<InkSplatGPU>.stride * arr.count, index: 1) }
        ce.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
    }
    let v0 = velCur, v1 = 1 - velCur, d0 = dyeCur, d1 = 1 - dyeCur
    run("ink_step", [vel[v0], dye[d0], vel[v1], dye[d1]], splats: sp); dyeCur = d1
    run("ink_vorticity", [vel[v1], vel[v0]])
    run("ink_divergence", [vel[v0], div])
    for _ in 0..<jacobiIters { let p0 = prCur, p1 = 1 - prCur; run("ink_jacobi", [pr[p0], div, pr[p1]]); prCur = p1 }
    run("ink_project", [vel[v0], pr[prCur], vel[v1]]); velCur = v1
    ce.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
    let ms = (cmd.gpuEndTime - cmd.gpuStartTime) * 1000
    if paceHz > 0 { let next = paceStart + UInt64(Double(n + 1) / paceHz * 1e9); let now = DispatchTime.now().uptimeNanoseconds
        if next > now { usleep(UInt32((next - now) / 1000)) } }
    gpuCsv += "\(n),\(sp.count),\(String(format: "%.4f", ms))\n"
    // record the splats of this step so any port can replay the script
    manifest.append(["step": n, "splats": sp.map { ["posVel": [$0.posVel.x, $0.posVel.y, $0.posVel.z, $0.posVel.w],
                                                     "color": [$0.color.x, $0.color.y, $0.color.z],
                                                     "params": [$0.params.x, $0.params.y, $0.params.z]] }])
    if (n + 1) % dumpEvery == 0 || n == steps - 1 {
        let tag = String(format: "%04d", n + 1)
        dump("vel_\(tag).f32", readTex(vel[velCur], channels: 2, stride: 2))
        dump("dye_\(tag).f32", readTex(dye[dyeCur], channels: 3, stride: 4))
        dump("pr_\(tag).f32", readTex(pr[prCur], channels: 1, stride: 1))
    }
}
try! gpuCsv.write(toFile: "\(outDir)/gpu_ms.csv", atomically: true, encoding: .utf8)
let info: [String: Any] = ["device": device.name, "storage": half ? "fp16 textures (shipped formats)" : "fp32 textures", "source": args.count > 5 ? args[5] : "synthetic", "W": W, "H": H, "dt": dt, "velDamp": velDamp, "dyeDamp": dyeDamp,
    "vorticity": vorticity, "jacobiIters": jacobiIters, "steps": steps, "dumpEvery": dumpEvery, "seed": seed,
    "layout": "row-major (y, x); vel rg, dye rgb, pr r; float32 LE", "script": manifest]
try! JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: "\(outDir)/manifest.json"))
print("trace: \(steps) steps on \(device.name), dumps every \(dumpEvery) -> \(outDir)")
