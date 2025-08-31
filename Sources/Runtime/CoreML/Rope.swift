#if canImport(CoreML)
import CoreML
import Foundation

/// Utilities for generating and applying rotary position embedding tables.
@available(iOS 15.0, macOS 12.0, *)
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
        
        var angles = [Float](repeating: 0, count: sequenceLength * halfDim)
        for pos in 0..<sequenceLength {
            for i in 0..<halfDim {
                let freq = 1.0 / pow(base, 2.0 * Float(i) / Float(headDimension))
                angles[pos * halfDim + i] = Float(pos) * freq
            }
        }
        var sinVals = [Float](repeating: 0, count: angles.count)
        var cosVals = [Float](repeating: 0, count: angles.count)
        for i in 0..<angles.count {
            sinVals[i] = Foundation.sin(angles[i])
            cosVals[i] = Foundation.cos(angles[i])
        }
        let shape: [Int] = [sequenceLength, halfDim]
        let sinArray = MLShapedArray<Float32>(scalars: sinVals, shape: shape)
        let cosArray = MLShapedArray<Float32>(scalars: cosVals, shape: shape)
        return (sinArray, cosArray)
    }
    
    /// Applies rotary position embeddings to a tensor using the "rotate half" method.
    /// - Parameters:
    ///   - tensor: Input tensor with shape [numberOfHeads, headDimension]
    ///   - sin: Sine values for the current position
    ///   - cos: Cosine values for the current position
    ///   - numberOfHeads: Number of attention heads
    /// - Returns: Rotated tensor with same shape as input
    public static func rotateHalf(
        _ tensor: MLMultiArray,
        sin: [Float32],
        cos: [Float32],
        numberOfHeads: Int
    ) throws -> MLMultiArray {
        let shape = tensor.shape.map { $0.intValue }
        let result = try MLMultiArray(shape: tensor.shape, dataType: tensor.dataType)
        let halfDim = shape.last! / 2
        
        // Rotate the tensor using RoPE formulation
        for headIdx in 0..<numberOfHeads {
            for i in 0..<halfDim {
                let x1Idx = [NSNumber(value: headIdx), NSNumber(value: i)]
                let x2Idx = [NSNumber(value: headIdx), NSNumber(value: i + halfDim)]
                
                let x1 = tensor[x1Idx].floatValue
                let x2 = tensor[x2Idx].floatValue
                
                result[x1Idx] = NSNumber(value: x1 * cos[i] - x2 * sin[i])
                result[x2Idx] = NSNumber(value: x1 * sin[i] + x2 * cos[i])
            }
        }
        
        return result
    }
}
#endif
