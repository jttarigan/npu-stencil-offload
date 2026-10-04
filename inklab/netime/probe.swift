import Foundation
import CoreML
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
let m = try MLModel(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]), configuration: cfg)
var inputs: [String: MLFeatureValue] = [:]
for (name, d) in m.modelDescription.inputDescriptionsByName {
    let a = try MLMultiArray(shape: d.multiArrayConstraint!.shape, dataType: .float32); inputs[name] = MLFeatureValue(multiArray: a) }
let out = try m.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
for name in ["vel_out", "dye_out", "pr_out"] {
    let a = out.featureValue(for: name)!.multiArrayValue!
    print(name, "type", a.dataType.rawValue, "shape", a.shape, "strides", a.strides, "count", a.count)
}
