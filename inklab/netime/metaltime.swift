// The GPU baseline on the Mac: K simulation steps per command buffer, then
// waitUntilCompleted, timed as wall clock per submission (the same protocol as
// netime and the Android GLES path). The ink kernels are read verbatim out of
// ../ink_kernels.metal (the game's kernels) at run time.
//   metaltime <ink|heat|grayscott> <sweeps> <K> [calls=300] [hz=0]
import Foundation
import Metal

let a = CommandLine.arguments
let workload = a[1], sweeps = Int(a[2])!, K = Int(a[3])!
let calls = a.count > 4 ? Int(a[4])! : 300
let hz = a.count > 5 ? Double(a[5])! : 0
let W = 128, H = 256, jacobi = 26

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!

// --- kernels -------------------------------------------------------------
let here = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let game = try String(contentsOf: here.appendingPathComponent("../ink_kernels.metal"))
func slice(_ s: String, from: String, through: String) -> String {
    let lo = s.range(of: from)!.lowerBound
    let tail = s[lo...]
    let hi = tail.range(of: through)!.upperBound
    let close = tail[hi...].range(of: "\n}\n")!.upperBound       // end of that kernel
    return String(tail[..<close])
}
let inkSrc = slice(game, from: "struct InkSplatG", through: "kernel void ink_zero")
let stencilSrc = """
kernel void heat(texture2d<float, access::read> fIn [[texture(0)]],
                 texture2d<float, access::write> fOut [[texture(1)]],
                 uint2 gid [[thread_position_in_grid]]) {
    int W = fIn.get_width(), H = fIn.get_height();
    if (int(gid.x) >= W || int(gid.y) >= H) return;
    int2 p = int2(gid);
    float c = fIn.read(gid).x;
    float lap = fIn.read(uint2(max(p.x - 1, 0), p.y)).x + fIn.read(uint2(min(p.x + 1, W - 1), p.y)).x
              + fIn.read(uint2(p.x, max(p.y - 1, 0))).x + fIn.read(uint2(p.x, min(p.y + 1, H - 1))).x - 4.0 * c;
    fOut.write(float4(c + 0.2 * lap, 0, 0, 0), gid);
}
kernel void grayscott(texture2d<float, access::read> fIn [[texture(0)]],
                      texture2d<float, access::write> fOut [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    int W = fIn.get_width(), H = fIn.get_height();
    if (int(gid.x) >= W || int(gid.y) >= H) return;
    int2 p = int2(gid);
    float2 c = fIn.read(gid).xy;
    float2 lap = fIn.read(uint2(max(p.x - 1, 0), p.y)).xy + fIn.read(uint2(min(p.x + 1, W - 1), p.y)).xy
               + fIn.read(uint2(p.x, max(p.y - 1, 0))).xy + fIn.read(uint2(p.x, min(p.y + 1, H - 1))).xy - 4.0 * c;
    float u = c.x, v = c.y, uvv = u * v * v;
    fOut.write(float4(u + 0.16 * lap.x - uvv + 0.035 * (1.0 - u),
                      v + 0.08 * lap.y + uvv - (0.035 + 0.065) * v, 0, 0), gid);
}
"""
let lib = try dev.makeLibrary(source: "#include <metal_stdlib>\nusing namespace metal;\n" + inkSrc + stencilSrc,
                              options: nil)
func pso(_ n: String) -> MTLComputePipelineState {
    try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!)
}

// --- state, formats as in the game (Renderer.inkTextures) -------------------
func tex(_ f: MTLPixelFormat) -> MTLTexture {
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: W, height: H, mipmapped: false)
    d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
    return dev.makeTexture(descriptor: d)!
}
struct InkU { var a: SIMD4<Float>; var b: SIMD4<Float> }
var u = InkU(a: SIMD4(1 / 30, 0.3, 0.03, 2), b: SIMD4(2.5, Float(W), Float(H), 0))
// Two splats per step, the same load as the Android GLES path.
let splats: [Float] = [40, 60, 20, -10, 0.9, 0.4, 0.2, 0, 3, 0.9, 0, 0,
                       90, 180, 0, 0, 1, 1, 1, 0, 4, 0.6, 5, 0]
let splatBuf = dev.makeBuffer(bytes: splats, length: splats.count * 4)!

var vel = [tex(.rg16Float), tex(.rg16Float)], dye = [tex(.rgba16Float), tex(.rgba16Float)]
var pr = [tex(.r16Float), tex(.r16Float)]; let div = tex(.r16Float)
var fld = [tex(workload == "heat" ? .r16Float : .rg16Float), tex(workload == "heat" ? .r16Float : .rg16Float)]
let tg = MTLSize(width: 8, height: 8, depth: 1)
let grid = MTLSize(width: (W + 7) / 8, height: (H + 7) / 8, depth: 1)

let psoStep = pso("ink_step"), psoVort = pso("ink_vorticity"), psoDiv = pso("ink_divergence")
let psoJac = pso("ink_jacobi"), psoProj = pso("ink_project")
let psoSt = workload == "ink" ? nil : pso(workload)

func encodeStep(_ e: MTLComputeCommandEncoder) {
    if workload == "ink" {
        e.setComputePipelineState(psoStep)
        e.setTexture(vel[0], index: 0); e.setTexture(dye[0], index: 1)
        e.setTexture(vel[1], index: 2); e.setTexture(dye[1], index: 3)
        e.setBytes(&u, length: MemoryLayout<InkU>.stride, index: 0); e.setBuffer(splatBuf, offset: 0, index: 1)
        e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        e.setComputePipelineState(psoVort)
        e.setTexture(vel[1], index: 0); e.setTexture(vel[0], index: 1); e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        e.setComputePipelineState(psoDiv)
        e.setTexture(vel[0], index: 0); e.setTexture(div, index: 1); e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        e.setComputePipelineState(psoJac)
        for _ in 0..<jacobi {
            e.setTexture(pr[0], index: 0); e.setTexture(div, index: 1); e.setTexture(pr[1], index: 2)
            e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg); pr.swapAt(0, 1)
        }
        e.setComputePipelineState(psoProj)
        e.setTexture(vel[0], index: 0); e.setTexture(pr[0], index: 1); e.setTexture(vel[1], index: 2)
        e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        vel.swapAt(0, 1); dye.swapAt(0, 1)
    } else {
        e.setComputePipelineState(psoSt!)
        for _ in 0..<sweeps {
            e.setTexture(fld[0], index: 0); e.setTexture(fld[1], index: 1)
            e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg); fld.swapAt(0, 1)
        }
    }
}

var gpuSec = 0.0                                     // GPU execution time, no submission/sync
func submit() {
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!          // serial dispatch type = implicit barriers, as in the game
    for _ in 0..<K { encodeStep(e) }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    gpuSec += cb.gpuEndTime - cb.gpuStartTime
}

for _ in 0..<20 { submit() }
gpuSec = 0
let t0 = DispatchTime.now().uptimeNanoseconds
var busy: UInt64 = 0
for i in 0..<calls {
    let s = DispatchTime.now().uptimeNanoseconds
    submit()
    busy += DispatchTime.now().uptimeNanoseconds - s
    if hz > 0 {
        let next = t0 + UInt64(Double(i + 1) / hz * 1e9)
        let now = DispatchTime.now().uptimeNanoseconds
        if next > now { usleep(UInt32((next - now) / 1000)) }
    }
}
// Both numbers are per SUBMISSION (K steps), like netime. gpu = execution time
// alone, i.e. the marginal cost when the work rides in an existing frame's buffer.
let gpuMs = gpuSec * 1e3 / Double(calls)
if hz > 0 { print(String(format: "paced at %.2f Hz: %.3f ms busy per step gpu %.3f", hz, Double(busy) / 1e6 / Double(calls), gpuMs)); exit(0) }
print(String(format: "metal %@ s%d u%d %.3f ms/step over %d calls gpu %.3f", workload, sweeps, K,
             Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 / Double(calls), calls, gpuMs))
