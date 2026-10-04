// Times a compiled Core ML ink-step model with preallocated inputs, so the
// number is the framework + accelerator cost per step without Python.
//   netime <model.mlmodelc> <cpu|gpu|ne|all> [calls=1000]
import Foundation
import CoreML

let args = CommandLine.arguments
let url = URL(fileURLWithPath: args[1])
let unitsArg = args.count > 2 ? args[2] : "ne"
let calls = args.count > 3 ? Int(args[3])! : 1000
let hz = args.count > 4 ? Double(args[4])! : 0     // paced mode: steps per second (0 = back to back)
let cfg = MLModelConfiguration()
switch unitsArg {
case "cpu": cfg.computeUnits = .cpuOnly
case "gpu": cfg.computeUnits = .cpuAndGPU
case "ne":  cfg.computeUnits = .cpuAndNeuralEngine
default:    cfg.computeUnits = .all
}
let model = try MLModel(contentsOf: url, configuration: cfg)
var inputs: [String: MLFeatureValue] = [:]
for (name, desc) in model.modelDescription.inputDescriptionsByName {
    let shape = desc.multiArrayConstraint!.shape
    let arr = try MLMultiArray(shape: shape, dataType: .float32)
    let n = arr.count; let p = arr.dataPointer.bindMemory(to: Float.self, capacity: n)
    for i in 0..<n { p[i] = Float.random(in: -1...1) * 0.1 }
    inputs[name] = MLFeatureValue(multiArray: arr)
}
let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
let opts = MLPredictionOptions()
for _ in 0..<20 { _ = try model.prediction(from: provider, options: opts) }
let t0 = DispatchTime.now().uptimeNanoseconds
var busyNs: UInt64 = 0
for i in 0..<calls {
    let a = DispatchTime.now().uptimeNanoseconds
    _ = try model.prediction(from: provider, options: opts)
    busyNs += DispatchTime.now().uptimeNanoseconds - a
    if hz > 0 {   // sleep until the next tick of the fixed cadence
        let next = t0 + UInt64(Double(i + 1) / hz * 1e9)
        let now = DispatchTime.now().uptimeNanoseconds
        if next > now { usleep(UInt32((next - now) / 1000)) }
    }
}
if hz > 0 { print(String(format: "paced at %.0f Hz: %.3f ms busy per step", hz, Double(busyNs) / 1e6 / Double(calls))); exit(0) }
let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 / Double(calls)
print(String(format: "%@ %@ %.3f ms/step over %d calls", url.lastPathComponent, unitsArg, ms, calls))
