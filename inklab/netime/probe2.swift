import Foundation
import CoreML
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuOnly
let m = try MLModel(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]), configuration: cfg)
let W = 128, H = 256, n = W * H
func arr(_ c: Int) -> MLMultiArray { let a = try! MLMultiArray(shape: [1, NSNumber(value: c), NSNumber(value: H), NSNumber(value: W)], dataType: .float32); memset(a.dataPointer, 0, a.count * 4); return a }
let vel = arr(2), dye = arr(3), pr = arr(1), velAdd = arr(2), dyeAdd = arr(3)
let da = dyeAdd.dataPointer.bindMemory(to: Float.self, capacity: 3 * n)
for y in 100..<140 { for x in 40..<80 { let i = y * W + x; da[i] = 0.5; da[n + i] = 0.2 } }
for step in 0..<5 {
    let out = try m.prediction(from: MLDictionaryFeatureProvider(dictionary: ["vel": MLFeatureValue(multiArray: vel), "dye": MLFeatureValue(multiArray: dye), "pr": MLFeatureValue(multiArray: pr), "velAdd": MLFeatureValue(multiArray: velAdd), "dyeAdd": MLFeatureValue(multiArray: dyeAdd)]))
    for (name, dst) in [("vel_out", vel), ("dye_out", dye), ("pr_out", pr)] {
        let src = out.featureValue(for: name)!.multiArrayValue!
        let p = dst.dataPointer.bindMemory(to: Float.self, capacity: dst.count)
        if src.dataType == .float32 { memcpy(p, src.dataPointer, dst.count * 4) }
        else { let h = src.dataPointer.bindMemory(to: Float16.self, capacity: src.count); for i in 0..<dst.count { p[i] = Float(h[i]) } }
    }
    let d = dye.dataPointer.bindMemory(to: Float.self, capacity: 3 * n)
    var mx: Float = 0, sum: Float = 0, nan = 0
    for i in 0..<(3 * n) { if d[i].isNaN { nan += 1 } else { mx = max(mx, d[i]); sum += d[i] } }
    print("step \(step): dye max \(mx) sum \(sum) nan \(nan)  srcType \(out.featureValue(for: "dye_out")!.multiArrayValue!.dataType.rawValue)")
}
