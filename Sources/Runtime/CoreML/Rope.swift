#if canImport(CoreML)
import CoreML
import Foundation

/// Utilities for generating rotary position embedding tables.
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
        let shape: [Int] = [sequenceLength, halfDim]
        let sinArray = MLShapedArray<Float32>(scalars: sinVals, shape: shape)
        let cosArray = MLShapedArray<Float32>(scalars: cosVals, shape: shape)
        return (sinArray, cosArray)
    }
}
#endif
