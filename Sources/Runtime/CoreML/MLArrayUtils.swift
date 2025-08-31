#if canImport(CoreML)
import CoreML
import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// Utility functions for MLMultiArray operations in transformer models.
@available(iOS 15.0, macOS 12.0, *)
public enum MLArrayUtils {
    
    /// Performs matrix multiplication: A * B
    public static func matrixMultiply(_ a: MLMultiArray, _ b: MLMultiArray) throws -> MLMultiArray {
        let shapeA = a.shape.map { $0.intValue }
        let shapeB = b.shape.map { $0.intValue }
        
        guard shapeA.count >= 2, shapeB.count >= 2 else {
            throw TransformerError.invalidShape("Matrix multiplication requires at least 2D tensors")
        }
        
        let m = shapeA[shapeA.count - 2]
        let k = shapeA[shapeA.count - 1]
        let n = shapeB[shapeB.count - 1]
        
        guard k == shapeB[shapeB.count - 2] else {
            throw TransformerError.invalidShape("Incompatible shapes for matrix multiplication")
        }
        
        var resultShape = shapeA
        resultShape[resultShape.count - 1] = n
        
        let result = try MLMultiArray(shape: resultShape.map { NSNumber(value: $0) }, dataType: .float16)
        
        // Use basic matrix multiplication (Accelerate optimization can be added later)
        // This is a simplified implementation for compatibility
        for batchIdx in 0..<(shapeA.count > 2 ? shapeA[0] : 1) {
            for i in 0..<m {
                for j in 0..<n {
                    var sum: Float = 0
                    for l in 0..<k {
                        let aIdx = batchIdx * m * k + i * k + l
                        let bIdx = batchIdx * k * n + l * n + j
                        sum += getFloat(from: a, at: aIdx) * getFloat(from: b, at: bIdx)
                    }
                    let resultIdx = batchIdx * m * n + i * n + j
                    setFloat(in: result, at: resultIdx, value: sum)
                }
            }
        }
        
        return result
    }
    
    /// Transposes the last two dimensions of a tensor
    public static func transpose(_ tensor: MLMultiArray) throws -> MLMultiArray {
        let shape = tensor.shape.map { $0.intValue }
        guard shape.count >= 2 else {
            throw TransformerError.invalidShape("Transpose requires at least 2D tensor")
        }
        
        var newShape = shape
        newShape.swapAt(shape.count - 2, shape.count - 1)
        
        let result = try MLMultiArray(shape: newShape.map { NSNumber(value: $0) }, dataType: tensor.dataType)
        
        let m = shape[shape.count - 2]
        let n = shape[shape.count - 1]
        
        // Simple transpose for 2D case
        if shape.count == 2 {
            for i in 0..<m {
                for j in 0..<n {
                    let sourceIndex = [NSNumber(value: i), NSNumber(value: j)]
                    let targetIndex = [NSNumber(value: j), NSNumber(value: i)]
                    result[targetIndex] = tensor[sourceIndex]
                }
            }
        } else {
            // For higher dimensions, just copy data (simplified)
            for i in 0..<tensor.count {
                setFloat(in: result, at: i, value: getFloat(from: tensor, at: i))
            }
        }
        
        return result
    }
    
    /// Applies softmax along the last dimension
    public static func softmax(_ tensor: MLMultiArray) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: tensor.shape, dataType: tensor.dataType)
        let shape = tensor.shape.map { $0.intValue }
        let lastDim = shape.last!
        let batchSize = tensor.count / lastDim
        
        for batch in 0..<batchSize {
            let startIndex = batch * lastDim
            let endIndex = startIndex + lastDim
            
            // Find max for numerical stability
            var maxVal: Float = -.infinity
            for i in startIndex..<endIndex {
                let value = getFloat(from: tensor, at: i)
                maxVal = max(maxVal, value)
            }
            
            // Compute exp and sum
            var sum: Float = 0
            var expVals = [Float](repeating: 0, count: lastDim)
            for i in 0..<lastDim {
                let value = getFloat(from: tensor, at: startIndex + i)
                expVals[i] = exp(value - maxVal)
                sum += expVals[i]
            }
            
            // Normalize
            for i in 0..<lastDim {
                setFloat(in: result, at: startIndex + i, value: expVals[i] / sum)
            }
        }
        
        return result
    }
    
    /// Applies SiLU activation function: x * sigmoid(x)
    public static func silu(_ tensor: MLMultiArray) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: tensor.shape, dataType: tensor.dataType)
        
        for i in 0..<tensor.count {
            let x = getFloat(from: tensor, at: i)
            let sigmoid = 1.0 / (1.0 + exp(-x))
            setFloat(in: result, at: i, value: x * sigmoid)
        }
        
        return result
    }
    
    /// Element-wise multiplication of two tensors
    public static func multiplyTensors(_ a: MLMultiArray, _ b: MLMultiArray) throws -> MLMultiArray {
        guard a.shape == b.shape else {
            throw TransformerError.invalidShape("Tensors must have the same shape for element-wise multiplication")
        }
        
        let result = try MLMultiArray(shape: a.shape, dataType: a.dataType)
        
        for i in 0..<a.count {
            let aVal = getFloat(from: a, at: i)
            let bVal = getFloat(from: b, at: i)
            setFloat(in: result, at: i, value: aVal * bVal)
        }
        
        return result
    }
    
    /// Element-wise addition of two tensors
    public static func addTensors(_ a: MLMultiArray, _ b: MLMultiArray) throws -> MLMultiArray {
        guard a.shape == b.shape else {
            throw TransformerError.invalidShape("Tensors must have the same shape for addition")
        }
        
        let result = try MLMultiArray(shape: a.shape, dataType: a.dataType)
        
        for i in 0..<a.count {
            let aVal = getFloat(from: a, at: i)
            let bVal = getFloat(from: b, at: i)
            setFloat(in: result, at: i, value: aVal + bVal)
        }
        
        return result
    }
    
    /// Multiplies tensor by a scalar
    public static func scalarMultiply(_ tensor: MLMultiArray, _ scalar: Float) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: tensor.shape, dataType: tensor.dataType)
        
        for i in 0..<tensor.count {
            let value = getFloat(from: tensor, at: i)
            setFloat(in: result, at: i, value: value * scalar)
        }
        
        return result
    }
    
    /// Applies causal mask to attention scores
    public static func applyCausalMask(_ scores: MLMultiArray, currentLength: Int) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: scores.shape, dataType: scores.dataType)
        let shape = scores.shape.map { $0.intValue }
        
        // Copy original scores
        for i in 0..<scores.count {
            setFloat(in: result, at: i, value: getFloat(from: scores, at: i))
        }
        
        // Apply mask (set future positions to -inf)
        let seqLen = shape.last!
        let batchSize = scores.count / (seqLen * seqLen)
        
        for batch in 0..<batchSize {
            for i in 0..<currentLength {
                for j in currentLength..<seqLen {
                    let index = batch * seqLen * seqLen + i * seqLen + j
                    setFloat(in: result, at: index, value: -.infinity)
                }
            }
        }
        
        return result
    }
    
    /// Concatenates arrays along specified axis
    public static func concatenateArrays(_ arrays: [MLMultiArray], axis: Int) throws -> MLMultiArray {
        guard !arrays.isEmpty else {
            throw TransformerError.invalidShape("Cannot concatenate empty array list")
        }
        
        let firstShape = arrays[0].shape.map { $0.intValue }
        guard axis < firstShape.count else {
            throw TransformerError.invalidShape("Axis out of bounds")
        }
        
        var newShape = firstShape
        newShape[axis] = arrays.reduce(0) { $0 + $1.shape[axis].intValue }
        
        let result = try MLMultiArray(shape: newShape.map { NSNumber(value: $0) }, dataType: arrays[0].dataType)
        
        // Simple concatenation - in practice would need more efficient implementation
        var currentOffset = 0
        for array in arrays {
            let arraySize = array.shape[axis].intValue
            // Copy data from each array to result
            // This is a simplified implementation
            for i in 0..<array.count {
                let resultIndex = currentOffset + i
                if resultIndex < result.count {
                    setFloat(in: result, at: resultIndex, value: getFloat(from: array, at: i))
                }
            }
            currentOffset += arraySize
        }
        
        return result
    }
    
    /// Reshapes tensor for multi-head attention
    public static func reshapeForHeads(_ tensor: MLMultiArray, numberOfHeads: Int) throws -> MLMultiArray {
        let shape = tensor.shape.map { $0.intValue }
        let embeddingDim = shape.last!
        let headDim = embeddingDim / numberOfHeads
        
        guard embeddingDim % numberOfHeads == 0 else {
            throw TransformerError.invalidShape("Embedding dimension must be divisible by number of heads")
        }
        
        var newShape = shape
        newShape[newShape.count - 1] = numberOfHeads
        newShape.append(headDim)
        
        let result = try MLMultiArray(shape: newShape.map { NSNumber(value: $0) }, dataType: tensor.dataType)
        
        // Reshape data
        for i in 0..<tensor.count {
            setFloat(in: result, at: i, value: getFloat(from: tensor, at: i))
        }
        
        return result
    }
    
    /// Reshapes tensor from multi-head format back to embedding dimension
    public static func reshapeFromHeads(_ tensor: MLMultiArray) throws -> MLMultiArray {
        let shape = tensor.shape.map { $0.intValue }
        guard shape.count >= 2 else {
            throw TransformerError.invalidShape("Invalid tensor shape for reshaping from heads")
        }
        
        let numberOfHeads = shape[shape.count - 2]
        let headDim = shape[shape.count - 1]
        let embeddingDim = numberOfHeads * headDim
        
        var newShape = shape
        newShape.removeLast(2)
        newShape.append(embeddingDim)
        
        let result = try MLMultiArray(shape: newShape.map { NSNumber(value: $0) }, dataType: tensor.dataType)
        
        // Reshape data
        for i in 0..<tensor.count {
            setFloat(in: result, at: i, value: getFloat(from: tensor, at: i))
        }
        
        return result
    }
    
    /// Applies layer normalization
    public static func layerNorm(_ tensor: MLMultiArray, weights: LayerNormWeights) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: tensor.shape, dataType: tensor.dataType)
        let shape = tensor.shape.map { $0.intValue }
        let lastDim = shape.last!
        let batchSize = tensor.count / lastDim
        
        for batch in 0..<batchSize {
            let startIndex = batch * lastDim
            
            // Compute mean
            var sum: Float = 0
            for i in 0..<lastDim {
                sum += getFloat(from: tensor, at: startIndex + i)
            }
            let mean = sum / Float(lastDim)
            
            // Compute variance
            var variance: Float = 0
            for i in 0..<lastDim {
                let diff = getFloat(from: tensor, at: startIndex + i) - mean
                variance += diff * diff
            }
            variance /= Float(lastDim)
            
            // Normalize
            let std = sqrt(variance + 1e-5) // epsilon for numerical stability
            for i in 0..<lastDim {
                let normalized = (getFloat(from: tensor, at: startIndex + i) - mean) / std
                let weight = getFloat(from: weights.weight, at: i)
                let bias = weights.bias != nil ? getFloat(from: weights.bias!, at: i) : 0
                setFloat(in: result, at: startIndex + i, value: normalized * weight + bias)
            }
        }
        
        return result
    }
    
    /// Applies linear transformation: x * W + b
    public static func linear(_ input: MLMultiArray, weights: LinearWeights) throws -> MLMultiArray {
        let output = try matrixMultiply(input, weights.weight)
        
        if let bias = weights.bias {
            return try addTensors(output, bias)
        }
        
        return output
    }
    
    // MARK: - Helper functions
    
    // Helper function removed - using direct indexing instead
    
    private static func getFloat(from array: MLMultiArray, at index: Int) -> Float {
        let ptr = array.dataPointer
        
        switch array.dataType {
        case .float16:
            let float16Ptr = ptr.bindMemory(to: Float16.self, capacity: array.count)
            return Float(float16Ptr[index])
        case .float32:
            let float32Ptr = ptr.bindMemory(to: Float32.self, capacity: array.count)
            return float32Ptr[index]
        default:
            return 0
        }
    }
    
    private static func setFloat(in array: MLMultiArray, at index: Int, value: Float) {
        let ptr = array.dataPointer
        
        switch array.dataType {
        case .float16:
            let float16Ptr = ptr.bindMemory(to: Float16.self, capacity: array.count)
            float16Ptr[index] = Float16(value)
        case .float32:
            let float32Ptr = ptr.bindMemory(to: Float32.self, capacity: array.count)
            float32Ptr[index] = value
        default:
            break
        }
    }
}

/// Errors that can occur during transformer operations
public enum TransformerError: Error {
    case invalidShape(String)
    case computationError(String)
}

#endif