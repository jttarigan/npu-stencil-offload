// Where Core ML places each operation of a compiled model (MLComputePlan,
// macOS 14.4+ / iOS 17.4+), so a "neural engine" timing can be checked against
// what actually ran there. InkBench carries the same function (placementRow)
// and writes one row per model before its sweep.
//
//   swiftc -O placement.swift -o placement
//   ./placement <model.mlmodelc>... [ne|all]      (default ne = cpuAndNeuralEngine)
//
// One CSV row per model: operation counts and Core ML's estimated cost share per
// preferred device, plus the op types that did not land on the neural engine.
import Foundation
import CoreML

@available(macOS 14.4, iOS 17.4, *)
func placementRow(_ url: URL, units: MLComputeUnits) async throws -> String {
    let cfg = MLModelConfiguration(); cfg.computeUnits = units
    let plan = try await MLComputePlan.load(contentsOf: url, configuration: cfg)
    guard case .program(let program) = plan.modelStructure, let main = program.functions["main"] else {
        return "\(url.deletingPathExtension().lastPathComponent),not an ML program"
    }
    var ops = ["ne": 0, "cpu": 0, "gpu": 0, "none": 0]
    var cost = ["ne": 0.0, "cpu": 0.0, "gpu": 0.0, "none": 0.0]
    var off: [String: Int] = [:]   // op type -> count, for ops not on the neural engine
    func walk(_ block: MLModelStructure.Program.Block) {
        for op in block.operations {
            for b in op.blocks { walk(b) }
            if op.operatorName == "const" { continue }   // weights, not work
            var d = "none"
            if let u = plan.deviceUsage(for: op) {
                switch u.preferred {
                case .neuralEngine: d = "ne"
                case .gpu: d = "gpu"
                case .cpu: d = "cpu"
                @unknown default: d = "none"
                }
            }
            ops[d]! += 1
            cost[d]! += plan.estimatedCost(of: op)?.weight ?? 0
            if d != "ne" { off[op.operatorName, default: 0] += 1 }
        }
    }
    walk(main.block)
    let total = max(cost.values.reduce(0, +), 1e-12)
    let offs = off.sorted { $0.value > $1.value }.map { "\($0.key):\($0.value)" }.joined(separator: " ")
    return [url.deletingPathExtension().lastPathComponent,
            "\(ops["ne"]!)", "\(ops["cpu"]!)", "\(ops["gpu"]!)", "\(ops["none"]!)",
            String(format: "%.3f", cost["ne"]! / total), String(format: "%.3f", cost["cpu"]! / total),
            String(format: "%.3f", cost["gpu"]! / total), offs].joined(separator: ",")
}

let placementHeader = "model,ops_ne,ops_cpu,ops_gpu,ops_unplaced,cost_ne,cost_cpu,cost_gpu,off_ne_op_types"

#if !INKBENCH
let args = CommandLine.arguments.dropFirst()
let units: MLComputeUnits = args.last == "all" ? .all : .cpuAndNeuralEngine
let paths = args.filter { $0 != "ne" && $0 != "all" }
let sem = DispatchSemaphore(value: 0)
Task {
    print(placementHeader)
    for p in paths {
        do { print(try await placementRow(URL(fileURLWithPath: p), units: units)) }
        catch { print("\(p),ERROR \(error)") }
    }
    sem.signal()
}
sem.wait()
#endif
