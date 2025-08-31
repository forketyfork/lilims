#if canImport(CoreML)
import CoreML
import Foundation
#if canImport(MetalPerformanceShadersGraph)
import MetalPerformanceShadersGraph
#endif

/// Utilities for generating rotary position embedding tables.
/// Uses MPSGraph when available, otherwise falls back to a pure Swift implementation.
public enum Rope {
    /// Generates sine and cosine tables for rotary position embeddings.
    /// - Parameters:
    ///   - sequenceLength: Maximum sequence length.
    ///   - headDimension: Dimension per attention head (must be even).
    ///   - base: Base frequency for rotary embeddings, default is 10k.
    /// - Returns: Tuple containing sine and cosine tables shaped `[sequenceLength, headDimension/2]`.
    public static func rotaryTables(sequenceLength: Int,
                                    headDimension: Int,
                                    base: Float = 10_000) -> (sine: MLShapedArray<Float32>, cosine: MLShapedArray<Float32>) {
        precondition(headDimension % 2 == 0, "headDimension must be even")
        let halfDim = headDimension / 2
#if canImport(MetalPerformanceShadersGraph)
        let graph = MPSGraph()
        let pos = graph.range(start: 0,
                              end: NSNumber(value: sequenceLength),
                              step: 1,
                              name: nil)
        let posReshaped = graph.reshape(pos, shape: [sequenceLength, 1], name: nil)
        let dims = graph.range(start: 0,
                               end: NSNumber(value: halfDim),
                               step: 1,
                               name: nil)
        let dimReshaped = graph.reshape(dims, shape: [1, NSNumber(value: halfDim)], name: nil)
        let exponent = graph.division(dimReshaped,
                                      graph.constant(NSNumber(value: halfDim), dataType: .float32),
                                      name: nil)
        let denom = graph.pow(graph.constant(NSNumber(value: base), dataType: .float32),
                              exponent,
                              name: nil)
        let invFreq = graph.division(graph.constant(1.0, dataType: .float32), denom, name: nil)
        let angles = graph.multiply(posReshaped, invFreq, name: nil)
        let sinTensor = graph.sin(angles, name: nil)
        let cosTensor = graph.cos(angles, name: nil)
        let results = graph.run(feeds: [:], targetTensors: [sinTensor, cosTensor], targetOperations: nil)
        let sinData = results[sinTensor]!
        let cosData = results[cosTensor]!
        let sinArray = MLShapedArray<Float32>(sinData.mltensorValue()!)
        let cosArray = MLShapedArray<Float32>(cosData.mltensorValue()!)
        return (sinArray, cosArray)
#else
        // Pure Swift fallback used when MPSGraph isn't available.
        var angles = [Float](repeating: 0, count: sequenceLength * halfDim)
        for pos in 0..<sequenceLength {
            for i in 0..<halfDim {
                let freq = pow(base, Float(i) / Float(halfDim))
                angles[pos * halfDim + i] = Float(pos) / freq
            }
        }
        var sinVals = [Float](repeating: 0, count: angles.count)
        var cosVals = [Float](repeating: 0, count: angles.count)
        for i in 0..<angles.count {
            sinVals[i] = Foundation.sin(angles[i])
            cosVals[i] = Foundation.cos(angles[i])
        }
        let shape: [NSNumber] = [NSNumber(value: sequenceLength), NSNumber(value: halfDim)]
        let sinArray = MLShapedArray<Float32>(scalars: sinVals, shape: shape)
        let cosArray = MLShapedArray<Float32>(scalars: cosVals, shape: shape)
        return (sinArray, cosArray)
#endif
    }
}
#endif
