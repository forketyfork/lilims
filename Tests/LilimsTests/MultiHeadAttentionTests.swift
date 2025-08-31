#if canImport(CoreML)
import XCTest
import CoreML
@testable import Lilims
@testable import RuntimeCoreML

/// Tests for Multi-Head Attention mechanism components.
@available(iOS 15.0, macOS 12.0, *)
final class MultiHeadAttentionTests: XCTestCase {
    
    private let config = TransformerConfig(
        vocabSize: 1000,
        maxSequenceLength: 512,
        embeddingDimension: 512,
        numberOfHeads: 8,
        numberOfLayers: 6
    )
    
    // MARK: - Projection Operations Tests
    
    func testQueryProjectionShape() throws {
        let batchSize = 1
        let seqLen = 1
        let embeddingDim = config.embeddingDimension
        
        // Create input tensor
        let inputShape = [batchSize, seqLen, embeddingDim].map { NSNumber(value: $0) }
        let input = try MLMultiArray(shape: inputShape, dataType: .float16)
        
        // Fill with test data
        for i in 0..<input.count {
            MLArrayUtilsTests.setFloat(in: input, at: i, value: Float.random(in: -1...1))
        }
        
        // Create projection weights
        let weightShape = [embeddingDim, embeddingDim].map { NSNumber(value: $0) }
        let weight = try MLMultiArray(shape: weightShape, dataType: .float16)
        let weights = LinearWeights(weight: weight, bias: nil)
        
        // Perform projection
        let result = try MLArrayUtils.linear(input, weights: weights)
        
        // Verify shape
        XCTAssertEqual(result.shape, input.shape, "Query projection should preserve input shape")
        XCTAssertEqual(result.shape.last?.intValue, embeddingDim, "Output should have embedding dimension")
    }
    
    func testKeyProjectionShape() throws {
        let batchSize = 1
        let seqLen = 1
        let embeddingDim = config.embeddingDimension
        
        // Create input tensor
        let inputShape = [batchSize, seqLen, embeddingDim].map { NSNumber(value: $0) }
        let input = try MLMultiArray(shape: inputShape, dataType: .float16)
        
        // Fill with test data
        for i in 0..<input.count {
            MLArrayUtilsTests.setFloat(in: input, at: i, value: Float.random(in: -1...1))
        }
        
        // Create projection weights
        let weightShape = [embeddingDim, embeddingDim].map { NSNumber(value: $0) }
        let weight = try MLMultiArray(shape: weightShape, dataType: .float16)
        let weights = LinearWeights(weight: weight, bias: nil)
        
        // Perform projection
        let result = try MLArrayUtils.linear(input, weights: weights)
        
        // Verify shape
        XCTAssertEqual(result.shape, input.shape, "Key projection should preserve input shape")
        XCTAssertEqual(result.shape.last?.intValue, embeddingDim, "Output should have embedding dimension")
    }
    
    func testValueProjectionShape() throws {
        let batchSize = 1
        let seqLen = 1
        let embeddingDim = config.embeddingDimension
        
        // Create input tensor
        let inputShape = [batchSize, seqLen, embeddingDim].map { NSNumber(value: $0) }
        let input = try MLMultiArray(shape: inputShape, dataType: .float16)
        
        // Fill with test data
        for i in 0..<input.count {
            MLArrayUtilsTests.setFloat(in: input, at: i, value: Float.random(in: -1...1))
        }
        
        // Create projection weights
        let weightShape = [embeddingDim, embeddingDim].map { NSNumber(value: $0) }
        let weight = try MLMultiArray(shape: weightShape, dataType: .float16)
        let weights = LinearWeights(weight: weight, bias: nil)
        
        // Perform projection
        let result = try MLArrayUtils.linear(input, weights: weights)
        
        // Verify shape
        XCTAssertEqual(result.shape, input.shape, "Value projection should preserve input shape")
        XCTAssertEqual(result.shape.last?.intValue, embeddingDim, "Output should have embedding dimension")
    }
    
    func testOutputProjectionBackToEmbeddingDimension() throws {
        let batchSize = 1
        let seqLen = 1
        let embeddingDim = config.embeddingDimension
        
        // Create input tensor (simulating output from multi-head attention)
        let inputShape = [batchSize, seqLen, embeddingDim].map { NSNumber(value: $0) }
        let input = try MLMultiArray(shape: inputShape, dataType: .float16)
        
        // Fill with test data
        for i in 0..<input.count {
            MLArrayUtilsTests.setFloat(in: input, at: i, value: Float.random(in: -1...1))
        }
        
        // Create projection weights (embedding_dim x embedding_dim)
        let weightShape = [embeddingDim, embeddingDim].map { NSNumber(value: $0) }
        let weight = try MLMultiArray(shape: weightShape, dataType: .float16)
        let weights = LinearWeights(weight: weight, bias: nil)
        
        // Perform output projection
        let result = try MLArrayUtils.linear(input, weights: weights)
        
        // Verify shape
        XCTAssertEqual(result.shape, input.shape, "Output projection should preserve tensor shape")
        XCTAssertEqual(result.shape.last?.intValue, embeddingDim, "Output should be back to embedding dimension")
    }
    
    func testProjectionWeightApplicationCorrectness() throws {
        let batchSize = 1
        let seqLen = 1
        let embeddingDim = 4 // Small dimension for easy verification
        
        // Create input tensor with known values
        let inputShape = [batchSize, seqLen, embeddingDim].map { NSNumber(value: $0) }
        let input = try MLMultiArray(shape: inputShape, dataType: .float16)
        
        // Set specific values [1, 2, 3, 4]
        for i in 0..<embeddingDim {
            MLArrayUtilsTests.setFloat(in: input, at: i, value: Float(i + 1))
        }
        
        // Create identity weight matrix
        let weightShape = [embeddingDim, embeddingDim].map { NSNumber(value: $0) }
        let weight = try MLMultiArray(shape: weightShape, dataType: .float16)
        
        // Initialize as identity matrix
        for i in 0..<embeddingDim {
            for j in 0..<embeddingDim {
                let index = i * embeddingDim + j
                MLArrayUtilsTests.setFloat(in: weight, at: index, value: i == j ? 1.0 : 0.0)
            }
        }
        
        let weights = LinearWeights(weight: weight, bias: nil)
        
        // Perform projection
        let result = try MLArrayUtils.linear(input, weights: weights)
        
        // Verify result (identity matrix should preserve input)
        for i in 0..<embeddingDim {
            let expected = Float(i + 1)
            let actual = MLArrayUtilsTests.getFloat(from: result, at: i)
            XCTAssertEqual(actual, expected, accuracy: 0.01, "Identity projection should preserve input values")
        }
    }
    
    // MARK: - Head Reshaping Tests
    
    func testReshapeForHeadsSplitsEmbeddingCorrectly() throws {
        let embeddingDim = config.embeddingDimension
        let numberOfHeads = config.numberOfHeads
        let expectedHeadDim = embeddingDim / numberOfHeads
        
        // Create tensor with embedding dimension
        let inputShape = [1, 1, embeddingDim].map { NSNumber(value: $0) }
        let input = try MLMultiArray(shape: inputShape, dataType: .float16)
        
        // Fill with sequential values
        for i in 0..<input.count {
            MLArrayUtilsTests.setFloat(in: input, at: i, value: Float(i))
        }
        
        // Reshape for heads
        let result = try MLArrayUtils.reshapeForHeads(input, numberOfHeads: numberOfHeads)
        
        // Check shape
        let expectedShape = [1, 1, numberOfHeads, expectedHeadDim].map { NSNumber(value: $0) }
        XCTAssertEqual(result.shape, expectedShape, "Reshape for heads should split embedding into heads and head dimension")
    }
    
    func testReshapeFromHeadsConcatenatesHeadsProperly() throws {
        let embeddingDim = config.embeddingDimension
        let numberOfHeads = config.numberOfHeads
        let headDim = embeddingDim / numberOfHeads
        
        // Create tensor in multi-head format
        let inputShape = [1, 1, numberOfHeads, headDim].map { NSNumber(value: $0) }
        let input = try MLMultiArray(shape: inputShape, dataType: .float16)
        
        // Fill with test data
        for i in 0..<input.count {
            MLArrayUtilsTests.setFloat(in: input, at: i, value: Float(i))
        }
        
        // Reshape from heads
        let result = try MLArrayUtils.reshapeFromHeads(input)
        
        // Check shape
        let expectedShape = [1, 1, embeddingDim].map { NSNumber(value: $0) }
        XCTAssertEqual(result.shape, expectedShape, "Reshape from heads should concatenate back to embedding dimension")
    }
    
    func testHeadDimensionCalculation() throws {
        let embeddingDim = config.embeddingDimension
        let numberOfHeads = config.numberOfHeads
        let expectedHeadDim = embeddingDim / numberOfHeads
        
        XCTAssertEqual(config.headDimension, expectedHeadDim, "Head dimension should be embedding dimension divided by number of heads")
        XCTAssertEqual(embeddingDim % numberOfHeads, 0, "Embedding dimension should be divisible by number of heads")
    }
    
    func testReshapingWithDifferentHeadCounts() throws {
        let embeddingDim = 64
        let testHeadCounts = [1, 2, 4, 8, 16]
        
        for numberOfHeads in testHeadCounts {
            let headDim = embeddingDim / numberOfHeads
            
            // Create tensor
            let inputShape = [1, 1, embeddingDim].map { NSNumber(value: $0) }
            let input = try MLMultiArray(shape: inputShape, dataType: .float16)
            
            // Reshape for heads
            let reshaped = try MLArrayUtils.reshapeForHeads(input, numberOfHeads: numberOfHeads)
            
            // Verify shape
            let expectedShape = [1, 1, numberOfHeads, headDim].map { NSNumber(value: $0) }
            XCTAssertEqual(reshaped.shape, expectedShape, "Reshaping should work with \(numberOfHeads) heads")
            
            // Reshape back
            let restored = try MLArrayUtils.reshapeFromHeads(reshaped)
            XCTAssertEqual(restored.shape, input.shape, "Reshaping back should restore original shape")
        }
    }
    
    func testReshapingPreservesDataOrdering() throws {
        let embeddingDim = 8 // Small for easy verification
        let numberOfHeads = 2
        
        // Create tensor with sequential values
        let inputShape = [1, 1, embeddingDim].map { NSNumber(value: $0) }
        let input = try MLMultiArray(shape: inputShape, dataType: .float16)
        
        for i in 0..<embeddingDim {
            MLArrayUtilsTests.setFloat(in: input, at: i, value: Float(i))
        }
        
        // Reshape for heads and back
        let reshaped = try MLArrayUtils.reshapeForHeads(input, numberOfHeads: numberOfHeads)
        let restored = try MLArrayUtils.reshapeFromHeads(reshaped)
        
        // Verify data is preserved
        for i in 0..<embeddingDim {
            let original = MLArrayUtilsTests.getFloat(from: input, at: i)
            let final = MLArrayUtilsTests.getFloat(from: restored, at: i)
            XCTAssertEqual(original, final, accuracy: 0.01, "Data should be preserved through reshape operations")
        }
    }
    
    // MARK: - Attention Computation Tests
    
    func testAttentionScoreCalculation() throws {
        let seqLen = 3
        let headDim = 4
        let numberOfHeads = 1
        
        // Create query and key tensors
        let qShape = [numberOfHeads, seqLen, headDim].map { NSNumber(value: $0) }
        let kShape = [numberOfHeads, seqLen, headDim].map { NSNumber(value: $0) }
        
        let queries = try MLMultiArray(shape: qShape, dataType: .float16)
        let keys = try MLMultiArray(shape: kShape, dataType: .float16)
        
        // Fill with known values for predictable results
        for i in 0..<queries.count {
            MLArrayUtilsTests.setFloat(in: queries, at: i, value: 1.0)
        }
        for i in 0..<keys.count {
            MLArrayUtilsTests.setFloat(in: keys, at: i, value: 0.5)
        }
        
        // Compute Q * K^T
        let keyTranspose = try MLArrayUtils.transpose(keys)
        let scores = try MLArrayUtils.matrixMultiply(queries, keyTranspose)
        
        // Verify shape
        let expectedShape = [numberOfHeads, seqLen, seqLen].map { NSNumber(value: $0) }
        XCTAssertEqual(scores.shape, expectedShape, "Attention scores should have shape [heads, seq_len, seq_len]")
        
        // Verify values (1.0 * 0.5 * headDim = 2.0)
        let expectedValue: Float = 1.0 * 0.5 * Float(headDim)
        for i in 0..<scores.count {
            let actual = MLArrayUtilsTests.getFloat(from: scores, at: i)
            XCTAssertEqual(actual, expectedValue, accuracy: 0.1, "Attention scores should be Q * K^T")
        }
    }
    
    func testAttentionScalingFactor() throws {
        let seqLen = 2
        let headDim = 4
        let numberOfHeads = 1
        
        // Create score tensor
        let scoreShape = [numberOfHeads, seqLen, seqLen].map { NSNumber(value: $0) }
        let scores = try MLMultiArray(shape: scoreShape, dataType: .float16)
        
        // Fill with known values
        for i in 0..<scores.count {
            MLArrayUtilsTests.setFloat(in: scores, at: i, value: Float(headDim)) // Use headDim value
        }
        
        // Apply scaling
        let scale = 1.0 / sqrt(Float(headDim))
        let scaledScores = try MLArrayUtils.scalarMultiply(scores, scale)
        
        // Verify scaling
        let expectedValue = Float(headDim) * scale
        for i in 0..<scaledScores.count {
            let actual = MLArrayUtilsTests.getFloat(from: scaledScores, at: i)
            XCTAssertEqual(actual, expectedValue, accuracy: 0.01, "Scores should be scaled by 1/sqrt(head_dim)")
        }
        
        // Verify the scale factor
        XCTAssertEqual(scale, 1.0 / sqrt(Float(headDim)), accuracy: 0.01, "Scale should be 1/sqrt(head_dim)")
    }
    
    func testCausalMaskApplication() throws {
        let seqLen = 3
        let numberOfHeads = 1
        let currentLength = 2
        
        // Create scores tensor
        let scoreShape = [numberOfHeads, seqLen, seqLen].map { NSNumber(value: $0) }
        let scores = try MLMultiArray(shape: scoreShape, dataType: .float16)
        
        // Fill with positive values
        for i in 0..<scores.count {
            MLArrayUtilsTests.setFloat(in: scores, at: i, value: 1.0)
        }
        
        // Apply causal mask
        let maskedScores = try MLArrayUtils.applyCausalMask(scores, currentLength: currentLength)
        
        // Verify that future positions are masked (set to -inf)
        // Check position [0, 2] should be masked (i=0, j=2, where j >= currentLength)
        let futurePositionIndex = 0 * seqLen * seqLen + 0 * seqLen + 2
        let maskedValue = MLArrayUtilsTests.getFloat(from: maskedScores, at: futurePositionIndex)
        XCTAssertTrue(maskedValue.isInfinite && maskedValue < 0, "Future positions should be masked with -inf")
        
        // Check that valid positions are not masked
        let validPositionIndex = 0 * seqLen * seqLen + 1 * seqLen + 0
        let validValue = MLArrayUtilsTests.getFloat(from: maskedScores, at: validPositionIndex)
        XCTAssertEqual(validValue, 1.0, accuracy: 0.01, "Valid positions should retain original values")
    }
    
    func testSoftmaxNormalizationOfAttentionWeights() throws {
        let seqLen = 3
        let numberOfHeads = 1
        
        // Create scores tensor
        let scoreShape = [numberOfHeads, seqLen, seqLen].map { NSNumber(value: $0) }
        let scores = try MLMultiArray(shape: scoreShape, dataType: .float16)
        
        // Fill with different values for each position
        let values: [Float] = [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.0, 1.0, 2.0]
        for i in 0..<min(scores.count, values.count) {
            MLArrayUtilsTests.setFloat(in: scores, at: i, value: values[i])
        }
        
        // Apply softmax
        let attentionWeights = try MLArrayUtils.softmax(scores)
        
        // Verify that each row sums to 1 (approximately)
        for row in 0..<seqLen {
            var rowSum: Float = 0
            for col in 0..<seqLen {
                let index = row * seqLen + col
                rowSum += MLArrayUtilsTests.getFloat(from: attentionWeights, at: index)
            }
            XCTAssertEqual(rowSum, 1.0, accuracy: 0.01, "Each attention row should sum to 1")
        }
        
        // Verify all values are positive
        for i in 0..<attentionWeights.count {
            let value = MLArrayUtilsTests.getFloat(from: attentionWeights, at: i)
            XCTAssertGreaterThanOrEqual(value, 0, "All attention weights should be non-negative")
        }
    }
    
    func testAttentionWeightApplicationToValues() throws {
        let seqLen = 2
        let headDim = 3
        let numberOfHeads = 1
        
        // Create attention weights (should sum to 1 per row)
        let weightShape = [numberOfHeads, seqLen, seqLen].map { NSNumber(value: $0) }
        let weights = try MLMultiArray(shape: weightShape, dataType: .float16)
        
        // Set attention weights: first row [0.6, 0.4], second row [0.3, 0.7]
        MLArrayUtilsTests.setFloat(in: weights, at: 0, value: 0.6) // [0,0]
        MLArrayUtilsTests.setFloat(in: weights, at: 1, value: 0.4) // [0,1]
        MLArrayUtilsTests.setFloat(in: weights, at: 2, value: 0.3) // [1,0]
        MLArrayUtilsTests.setFloat(in: weights, at: 3, value: 0.7) // [1,1]
        
        // Create values tensor
        let valueShape = [numberOfHeads, seqLen, headDim].map { NSNumber(value: $0) }
        let values = try MLMultiArray(shape: valueShape, dataType: .float16)
        
        // Set values: first position [1,2,3], second position [4,5,6]
        for i in 0..<headDim {
            MLArrayUtilsTests.setFloat(in: values, at: i, value: Float(i + 1))        // First position
            MLArrayUtilsTests.setFloat(in: values, at: headDim + i, value: Float(i + 4)) // Second position
        }
        
        // Apply attention: Attention * V
        let result = try MLArrayUtils.matrixMultiply(weights, values)
        
        // Verify shape
        let expectedShape = [numberOfHeads, seqLen, headDim].map { NSNumber(value: $0) }
        XCTAssertEqual(result.shape, expectedShape, "Result should have shape [heads, seq_len, head_dim]")
        
        // Verify values for first position: 0.6*[1,2,3] + 0.4*[4,5,6] = [2.2, 3.2, 4.2]
        let firstResult0 = MLArrayUtilsTests.getFloat(from: result, at: 0)
        let firstResult1 = MLArrayUtilsTests.getFloat(from: result, at: 1)
        let firstResult2 = MLArrayUtilsTests.getFloat(from: result, at: 2)
        
        XCTAssertEqual(firstResult0, 2.2, accuracy: 0.1, "Attention-weighted value should be correct")
        XCTAssertEqual(firstResult1, 3.2, accuracy: 0.1, "Attention-weighted value should be correct")
        XCTAssertEqual(firstResult2, 4.2, accuracy: 0.1, "Attention-weighted value should be correct")
    }
    
    func testAttentionWithSingleHead() throws {
        let seqLen = 2
        let headDim = 4
        let numberOfHeads = 1
        
        // Create Q, K, V tensors
        let shape = [numberOfHeads, seqLen, headDim].map { NSNumber(value: $0) }
        let queries = try MLMultiArray(shape: shape, dataType: .float16)
        let keys = try MLMultiArray(shape: shape, dataType: .float16)
        let values = try MLMultiArray(shape: shape, dataType: .float16)
        
        // Fill with test data
        for i in 0..<queries.count {
            MLArrayUtilsTests.setFloat(in: queries, at: i, value: 0.5)
            MLArrayUtilsTests.setFloat(in: keys, at: i, value: 0.5)
            MLArrayUtilsTests.setFloat(in: values, at: i, value: Float(i % 3))
        }
        
        // Compute attention
        let scale = 1.0 / sqrt(Float(headDim))
        let scores = try MLArrayUtils.matrixMultiply(queries, MLArrayUtils.transpose(keys))
        let scaledScores = try MLArrayUtils.scalarMultiply(scores, scale)
        let maskedScores = try MLArrayUtils.applyCausalMask(scaledScores, currentLength: seqLen)
        let attentionWeights = try MLArrayUtils.softmax(maskedScores)
        let result = try MLArrayUtils.matrixMultiply(attentionWeights, values)
        
        // Verify shape
        XCTAssertEqual(result.shape, shape, "Single head attention should preserve shape")
        
        // Verify result is finite and reasonable
        for i in 0..<result.count {
            let value = MLArrayUtilsTests.getFloat(from: result, at: i)
            XCTAssertTrue(value.isFinite, "Attention output should be finite")
        }
    }
    
    func testAttentionWithMultipleHeads() throws {
        let seqLen = 2
        let headDim = 4
        let numberOfHeads = 2
        
        // Create Q, K, V tensors
        let shape = [numberOfHeads, seqLen, headDim].map { NSNumber(value: $0) }
        let queries = try MLMultiArray(shape: shape, dataType: .float16)
        let keys = try MLMultiArray(shape: shape, dataType: .float16)
        let values = try MLMultiArray(shape: shape, dataType: .float16)
        
        // Fill with test data (different for each head)
        for h in 0..<numberOfHeads {
            for s in 0..<seqLen {
                for d in 0..<headDim {
                    let index = h * seqLen * headDim + s * headDim + d
                    MLArrayUtilsTests.setFloat(in: queries, at: index, value: Float(h + 1) * 0.5)
                    MLArrayUtilsTests.setFloat(in: keys, at: index, value: Float(h + 1) * 0.3)
                    MLArrayUtilsTests.setFloat(in: values, at: index, value: Float(d + 1))
                }
            }
        }
        
        // Compute attention for each head (simplified test)
        let scale = 1.0 / sqrt(Float(headDim))
        let scores = try MLArrayUtils.matrixMultiply(queries, MLArrayUtils.transpose(keys))
        let scaledScores = try MLArrayUtils.scalarMultiply(scores, scale)
        let maskedScores = try MLArrayUtils.applyCausalMask(scaledScores, currentLength: seqLen)
        let attentionWeights = try MLArrayUtils.softmax(maskedScores)
        let result = try MLArrayUtils.matrixMultiply(attentionWeights, values)
        
        // Verify shape
        XCTAssertEqual(result.shape, shape, "Multi-head attention should preserve shape")
        
        // Verify result is finite and reasonable for all heads
        for i in 0..<result.count {
            let value = MLArrayUtilsTests.getFloat(from: result, at: i)
            XCTAssertTrue(value.isFinite, "Multi-head attention output should be finite")
        }
    }
    
    func testAttentionNumericalStabilityWithLargeValues() throws {
        let seqLen = 2
        let numberOfHeads = 1
        
        // Create scores with large values that could cause overflow
        let scoreShape = [numberOfHeads, seqLen, seqLen].map { NSNumber(value: $0) }
        let scores = try MLMultiArray(shape: scoreShape, dataType: .float16)
        
        // Set large values
        for i in 0..<scores.count {
            MLArrayUtilsTests.setFloat(in: scores, at: i, value: 50.0) // Large value
        }
        
        // Apply softmax (should handle large values numerically stable)
        let attentionWeights = try MLArrayUtils.softmax(scores)
        
        // Verify that softmax output is stable
        for i in 0..<attentionWeights.count {
            let value = MLArrayUtilsTests.getFloat(from: attentionWeights, at: i)
            XCTAssertTrue(value.isFinite, "Softmax should be numerically stable with large inputs")
            XCTAssertGreaterThanOrEqual(value, 0, "Softmax output should be non-negative")
            XCTAssertLessThanOrEqual(value, 1, "Softmax output should be <= 1")
        }
        
        // Verify normalization (rows should sum to 1)
        for row in 0..<seqLen {
            var rowSum: Float = 0
            for col in 0..<seqLen {
                let index = row * seqLen + col
                rowSum += MLArrayUtilsTests.getFloat(from: attentionWeights, at: index)
            }
            XCTAssertEqual(rowSum, 1.0, accuracy: 0.01, "Softmax rows should sum to 1 even with large inputs")
        }
    }
}

// MARK: - Helper Extension

extension MultiHeadAttentionTests {
    
    /// Helper function to access MLArrayUtilsTests methods for setting/getting float values
    private struct MLArrayUtilsTests {
        static func setFloat(in array: MLMultiArray, at index: Int, value: Float) {
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
        
        static func getFloat(from array: MLMultiArray, at index: Int) -> Float {
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
    }
}

#endif