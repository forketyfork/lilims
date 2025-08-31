#if canImport(CoreML)
import XCTest
import CoreML
@testable import RuntimeCoreML

@available(iOS 15.0, macOS 12.0, *)
final class SimplifiedTransformerTests: XCTestCase {
    
    // MARK: - Configuration Tests
    
    func testTransformerConfig() {
        let config = TransformerConfig(
            vocabSize: 1000,
            maxSequenceLength: 512,
            embeddingDimension: 768,
            numberOfHeads: 12,
            numberOfLayers: 6,
            ropeBase: 10_000
        )
        
        XCTAssertEqual(config.vocabSize, 1000)
        XCTAssertEqual(config.maxSequenceLength, 512)
        XCTAssertEqual(config.embeddingDimension, 768)
        XCTAssertEqual(config.numberOfHeads, 12)
        XCTAssertEqual(config.numberOfLayers, 6)
        XCTAssertEqual(config.ropeBase, 10_000)
        XCTAssertEqual(config.headDimension, 64) // 768 / 12
    }
    
    // MARK: - StatefulTransformerModel Tests
    
    func testTransformerModelInitialization() {
        let config = createTestConfig()
        let model = StatefulTransformerModel(config: config)
        
        XCTAssertNotNil(model)
    }
    
    func testModelReset() {
        let config = createTestConfig()
        let model = StatefulTransformerModel(config: config)
        
        XCTAssertNoThrow(model.reset())
        
        // Multiple resets should be safe
        XCTAssertNoThrow(model.reset())
        XCTAssertNoThrow(model.reset())
    }
    
    // MARK: - RoPE Tests
    
    func testRotaryTablesBasic() {
        let (sineTable, cosineTable) = Rope.rotaryTables(
            sequenceLength: 4,
            headDimension: 4,
            base: 10_000
        )
        
        // Check shapes
        XCTAssertEqual(sineTable.shape.count, 2)
        XCTAssertEqual(sineTable.shape[0], 4) // sequenceLength
        XCTAssertEqual(sineTable.shape[1], 2) // headDimension/2
        XCTAssertEqual(cosineTable.shape[0], 4)
        XCTAssertEqual(cosineTable.shape[1], 2)
        
        // Check that we have the expected number of elements
        XCTAssertEqual(sineTable.scalars.count, 8) // 4 * 2
        XCTAssertEqual(cosineTable.scalars.count, 8)
    }
    
    func testRotaryTablesEvenHeadDimensionRequirement() {
        // Should work with even head dimension
        XCTAssertNoThrow({
            _ = Rope.rotaryTables(sequenceLength: 2, headDimension: 4)
        })
        
        // Note: precondition failures aren't catchable as errors in unit tests
        // So we just test the successful case for now
    }
    
    // MARK: - MLArrayUtils Basic Tests
    
    func testMLArrayUtilsMatrixMultiplicationShapes() throws {
        // Test that matrix multiplication can be called without errors
        let arrayA = try MLMultiArray(shape: [NSNumber(value: 2), NSNumber(value: 3)], dataType: .float16)
        let arrayB = try MLMultiArray(shape: [NSNumber(value: 3), NSNumber(value: 2)], dataType: .float16)
        
        // Fill with simple values using dataPointer
        fillArrayWithValue(arrayA, value: 1.0)
        fillArrayWithValue(arrayB, value: 2.0)
        
        let result = try MLArrayUtils.matrixMultiply(arrayA, arrayB)
        XCTAssertEqual(result.shape[0].intValue, 2)
        XCTAssertEqual(result.shape[1].intValue, 2)
    }
    
    func testMLArrayUtilsTranspose() throws {
        let array = try MLMultiArray(shape: [NSNumber(value: 2), NSNumber(value: 3)], dataType: .float16)
        
        let result = try MLArrayUtils.transpose(array)
        XCTAssertEqual(result.shape[0].intValue, 3)
        XCTAssertEqual(result.shape[1].intValue, 2)
    }
    
    func testMLArrayUtilsSoftmax() throws {
        let array = try MLMultiArray(shape: [NSNumber(value: 2), NSNumber(value: 3)], dataType: .float16)
        fillArrayWithValue(array, value: 1.0)
        
        let result = try MLArrayUtils.softmax(array)
        XCTAssertEqual(result.shape[0].intValue, 2)
        XCTAssertEqual(result.shape[1].intValue, 3)
    }
    
    func testMLArrayUtilsSiLU() throws {
        let array = try MLMultiArray(shape: [NSNumber(value: 4)], dataType: .float16)
        fillArrayWithValue(array, value: 1.0)
        
        let result = try MLArrayUtils.silu(array)
        XCTAssertEqual(result.shape[0].intValue, 4)
    }
    
    // MARK: - Weight Structure Tests
    
    func testLinearWeightsStructure() throws {
        let weight = try MLMultiArray(shape: [NSNumber(value: 10), NSNumber(value: 20)], dataType: .float16)
        let bias = try MLMultiArray(shape: [NSNumber(value: 1), NSNumber(value: 20)], dataType: .float16)
        
        let linearWeights = LinearWeights(weight: weight, bias: bias)
        
        XCTAssertEqual(linearWeights.weight.shape[0].intValue, 10)
        XCTAssertEqual(linearWeights.weight.shape[1].intValue, 20)
        XCTAssertNotNil(linearWeights.bias)
        XCTAssertEqual(linearWeights.bias?.shape[1].intValue, 20)
    }
    
    func testLayerNormWeightsStructure() throws {
        let weight = try MLMultiArray(shape: [NSNumber(value: 768)], dataType: .float16)
        let bias = try MLMultiArray(shape: [NSNumber(value: 768)], dataType: .float16)
        
        let layerNormWeights = LayerNormWeights(weight: weight, bias: bias)
        
        XCTAssertEqual(layerNormWeights.weight.shape[0].intValue, 768)
        XCTAssertNotNil(layerNormWeights.bias)
        XCTAssertEqual(layerNormWeights.bias?.shape[0].intValue, 768)
    }
    
    // MARK: - Error Handling Tests
    
    func testIncompatibleMatrixMultiplication() throws {
        let arrayA = try MLMultiArray(shape: [NSNumber(value: 2), NSNumber(value: 3)], dataType: .float16)
        let arrayB = try MLMultiArray(shape: [NSNumber(value: 4), NSNumber(value: 2)], dataType: .float16) // Incompatible: 3 != 4
        
        XCTAssertThrowsError(try MLArrayUtils.matrixMultiply(arrayA, arrayB)) { error in
            XCTAssertTrue(error is TransformerError)
            if case .invalidShape(let message) = error as! TransformerError {
                XCTAssertTrue(message.contains("Incompatible shapes"))
            }
        }
    }
    
    func testInvalidConfigurationHandling() {
        // Test configuration with incompatible head dimension
        let invalidConfig = TransformerConfig(
            vocabSize: 1000,
            maxSequenceLength: 512,
            embeddingDimension: 100, // Not perfectly divisible by numberOfHeads
            numberOfHeads: 12,
            numberOfLayers: 6
        )
        
        // This should create model but might have integer division
        let model = StatefulTransformerModel(config: invalidConfig)
        XCTAssertNotNil(model)
        
        // Head dimension should be calculated as 100/12 = 8 (integer division)
        XCTAssertEqual(invalidConfig.headDimension, 8)
    }
    
    // MARK: - Integration Tests
    
    func testMultiLayerModelConsistency() throws {
        let config = TransformerConfig(
            vocabSize: 100,
            maxSequenceLength: 8,
            embeddingDimension: 32,
            numberOfHeads: 4,
            numberOfLayers: 3
        )
        
        let model = StatefulTransformerModel(config: config)
        
        // Reset should work for all layers
        XCTAssertNoThrow(model.reset())
        
        // Multiple resets should be safe
        XCTAssertNoThrow(model.reset())
        XCTAssertNoThrow(model.reset())
    }
    
    func testRopeWithDifferentHeadDimensions() {
        // Test various head dimensions that are powers of 2
        let headDimensions = [32, 64, 128]
        
        for headDim in headDimensions {
            XCTAssertNoThrow({
                let (sine, cosine) = Rope.rotaryTables(
                    sequenceLength: 4,
                    headDimension: headDim
                )
                
                XCTAssertEqual(sine.shape[0], 4)
                XCTAssertEqual(sine.shape[1], headDim/2)
                XCTAssertEqual(cosine.shape[0], 4)
                XCTAssertEqual(cosine.shape[1], headDim/2)
            })
        }
    }
    
    // MARK: - Performance Tests
    
    func testRopePerformance() {
        // Test that RoPE generation is reasonably fast for typical model sizes
        measure {
            for _ in 0..<10 {
                _ = Rope.rotaryTables(
                    sequenceLength: 512,
                    headDimension: 64,
                    base: 10_000
                )
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func createTestConfig() -> TransformerConfig {
        return TransformerConfig(
            vocabSize: 100,
            maxSequenceLength: 64,
            embeddingDimension: 64, // Small for testing
            numberOfHeads: 4,
            numberOfLayers: 2
        )
    }
    
    private func fillArrayWithValue(_ array: MLMultiArray, value: Float) {
        // Fill array with a constant value using dataPointer
        for i in 0..<array.count {
            array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)[i] = Float16(value)
        }
    }
}

#endif