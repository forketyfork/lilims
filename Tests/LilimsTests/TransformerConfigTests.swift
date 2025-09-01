#if canImport(CoreML)
import XCTest
@testable import RuntimeCoreML

/// Tests for TransformerConfig structure - Part 1 of the comprehensive test suite
final class TransformerConfigTests: XCTestCase {
    
    func testValidConfigurationInitialization() throws {
        // Test configuration initialization with all parameters
        let config = TransformerConfig(
            vocabSize: 32000,
            maxSequenceLength: 2048,
            embeddingDimension: 512,
            numberOfHeads: 8,
            numberOfLayers: 6,
            ropeBase: 10000
        )
        
        XCTAssertEqual(config.vocabSize, 32000)
        XCTAssertEqual(config.maxSequenceLength, 2048)
        XCTAssertEqual(config.embeddingDimension, 512)
        XCTAssertEqual(config.numberOfHeads, 8)
        XCTAssertEqual(config.numberOfLayers, 6)
        XCTAssertEqual(config.ropeBase, 10000)
    }
    
    func testDefaultParameterValues() throws {
        // Test that ropeBase defaults to 10,000
        let config = TransformerConfig(
            vocabSize: 1000,
            maxSequenceLength: 128,
            embeddingDimension: 64,
            numberOfHeads: 4,
            numberOfLayers: 2
        )
        
        XCTAssertEqual(config.ropeBase, 10000, "ropeBase should default to 10,000")
    }
    
    func testHeadDimensionCalculation() throws {
        // Test headDimension calculation (embeddingDimension / numberOfHeads)
        let testCases = [
            (embeddingDim: 512, numberOfHeads: 8, expectedHeadDim: 64),
            (embeddingDim: 768, numberOfHeads: 12, expectedHeadDim: 64),
            (embeddingDim: 1024, numberOfHeads: 16, expectedHeadDim: 64),
            (embeddingDim: 256, numberOfHeads: 4, expectedHeadDim: 64),
            (embeddingDim: 128, numberOfHeads: 2, expectedHeadDim: 64)
        ]
        
        for testCase in testCases {
            let config = TransformerConfig(
                vocabSize: 1000,
                maxSequenceLength: 128,
                embeddingDimension: testCase.embeddingDim,
                numberOfHeads: testCase.numberOfHeads,
                numberOfLayers: 2
            )
            
            XCTAssertEqual(
                config.headDimension,
                testCase.expectedHeadDim,
                "Head dimension should be \(testCase.expectedHeadDim) for embedding dim \(testCase.embeddingDim) and \(testCase.numberOfHeads) heads"
            )
        }
    }
    
    func testEdgeCasesForEmbeddingDimensions() throws {
        // Test odd embedding dimensions - should still work with integer division
        let oddEmbeddingConfig = TransformerConfig(
            vocabSize: 1000,
            maxSequenceLength: 128,
            embeddingDimension: 63, // odd number
            numberOfHeads: 9,
            numberOfLayers: 2
        )
        
        // 63 / 9 = 7 (integer division)
        XCTAssertEqual(oddEmbeddingConfig.headDimension, 7)
        
        // Test very small embedding dimensions
        let smallEmbeddingConfig = TransformerConfig(
            vocabSize: 100,
            maxSequenceLength: 32,
            embeddingDimension: 8, // very small
            numberOfHeads: 2,
            numberOfLayers: 1
        )
        
        XCTAssertEqual(smallEmbeddingConfig.headDimension, 4)
        
        // Test single head case
        let singleHeadConfig = TransformerConfig(
            vocabSize: 1000,
            maxSequenceLength: 128,
            embeddingDimension: 64,
            numberOfHeads: 1,
            numberOfLayers: 2
        )
        
        XCTAssertEqual(singleHeadConfig.headDimension, 64)
    }
    
    func testConfigurationWithVariousVocabSizes() throws {
        // Test small vocab size
        let smallVocabConfig = TransformerConfig(
            vocabSize: 100,
            maxSequenceLength: 128,
            embeddingDimension: 64,
            numberOfHeads: 4,
            numberOfLayers: 2
        )
        XCTAssertEqual(smallVocabConfig.vocabSize, 100)
        
        // Test medium vocab size
        let mediumVocabConfig = TransformerConfig(
            vocabSize: 32000,
            maxSequenceLength: 1024,
            embeddingDimension: 512,
            numberOfHeads: 8,
            numberOfLayers: 6
        )
        XCTAssertEqual(mediumVocabConfig.vocabSize, 32000)
        
        // Test large vocab size
        let largeVocabConfig = TransformerConfig(
            vocabSize: 128000,
            maxSequenceLength: 2048,
            embeddingDimension: 1024,
            numberOfHeads: 16,
            numberOfLayers: 12
        )
        XCTAssertEqual(largeVocabConfig.vocabSize, 128000)
    }
    
    func testMaximumSequenceLengthBoundaries() throws {
        // Test very small sequence length
        let shortSeqConfig = TransformerConfig(
            vocabSize: 1000,
            maxSequenceLength: 1, // minimal sequence length
            embeddingDimension: 64,
            numberOfHeads: 4,
            numberOfLayers: 2
        )
        XCTAssertEqual(shortSeqConfig.maxSequenceLength, 1)
        
        // Test common sequence lengths
        let commonLengths = [128, 256, 512, 1024, 2048, 4096]
        for length in commonLengths {
            let config = TransformerConfig(
                vocabSize: 32000,
                maxSequenceLength: length,
                embeddingDimension: 512,
                numberOfHeads: 8,
                numberOfLayers: 6
            )
            XCTAssertEqual(config.maxSequenceLength, length)
        }
        
        // Test very large sequence length
        let longSeqConfig = TransformerConfig(
            vocabSize: 32000,
            maxSequenceLength: 32768, // very large
            embeddingDimension: 1024,
            numberOfHeads: 16,
            numberOfLayers: 12
        )
        XCTAssertEqual(longSeqConfig.maxSequenceLength, 32768)
    }
}

// MARK: - Configuration Validation Tests
extension TransformerConfigTests {
    
    func testConfigurationValidationForIncompatibleDimensions() throws {
        // The current implementation doesn't validate dimensions during initialization
        // These tests verify that we can create configurations with various dimension combinations
        // In a production system, we might want to add validation
        
        // Test when embedding dimension is not perfectly divisible by number of heads
        let config1 = TransformerConfig(
            vocabSize: 1000,
            maxSequenceLength: 128,
            embeddingDimension: 100, // not divisible by 8
            numberOfHeads: 8,
            numberOfLayers: 2
        )
        
        // This will result in headDimension = 12 (integer division: 100 / 8 = 12)
        XCTAssertEqual(config1.headDimension, 12)
        
        // Test extreme cases
        let config2 = TransformerConfig(
            vocabSize: 1,
            maxSequenceLength: 1,
            embeddingDimension: 1,
            numberOfHeads: 2, // This would create fractional head dimensions
            numberOfLayers: 1
        )
        
        // This will result in headDimension = 0 (integer division: 1 / 2 = 0)
        XCTAssertEqual(config2.headDimension, 0)
    }
}

// MARK: - Configuration Equality and Copy Tests  
extension TransformerConfigTests {
    
    func testConfigurationEquality() throws {
        // Since TransformerConfig is a struct, it automatically gets memberwise equality
        let config1 = TransformerConfig(
            vocabSize: 32000,
            maxSequenceLength: 2048,
            embeddingDimension: 512,
            numberOfHeads: 8,
            numberOfLayers: 6,
            ropeBase: 10000
        )
        
        let config2 = TransformerConfig(
            vocabSize: 32000,
            maxSequenceLength: 2048,
            embeddingDimension: 512,
            numberOfHeads: 8,
            numberOfLayers: 6,
            ropeBase: 10000
        )
        
        // Test that identical configurations are equal (if Equatable is implemented)
        // Note: Current implementation doesn't conform to Equatable
        // We test individual properties instead
        XCTAssertEqual(config1.vocabSize, config2.vocabSize)
        XCTAssertEqual(config1.maxSequenceLength, config2.maxSequenceLength)
        XCTAssertEqual(config1.embeddingDimension, config2.embeddingDimension)
        XCTAssertEqual(config1.numberOfHeads, config2.numberOfHeads)
        XCTAssertEqual(config1.numberOfLayers, config2.numberOfLayers)
        XCTAssertEqual(config1.ropeBase, config2.ropeBase)
        XCTAssertEqual(config1.headDimension, config2.headDimension)
    }
    
    func testConfigurationCopy() throws {
        // Test that struct copying works correctly
        let originalConfig = TransformerConfig(
            vocabSize: 32000,
            maxSequenceLength: 2048,
            embeddingDimension: 512,
            numberOfHeads: 8,
            numberOfLayers: 6,
            ropeBase: 5000
        )
        
        // Copy the configuration
        let copiedConfig = originalConfig
        
        // Verify all properties are copied correctly
        XCTAssertEqual(originalConfig.vocabSize, copiedConfig.vocabSize)
        XCTAssertEqual(originalConfig.maxSequenceLength, copiedConfig.maxSequenceLength)
        XCTAssertEqual(originalConfig.embeddingDimension, copiedConfig.embeddingDimension)
        XCTAssertEqual(originalConfig.numberOfHeads, copiedConfig.numberOfHeads)
        XCTAssertEqual(originalConfig.numberOfLayers, copiedConfig.numberOfLayers)
        XCTAssertEqual(originalConfig.ropeBase, copiedConfig.ropeBase)
        XCTAssertEqual(originalConfig.headDimension, copiedConfig.headDimension)
        
        // Since TransformerConfig is a struct, modifications to the copy don't affect the original
        // (We can't actually modify the struct since all properties are let constants)
    }
}

#endif