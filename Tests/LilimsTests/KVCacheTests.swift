#if canImport(CoreML)
import XCTest
import CoreML
import Foundation
@testable import RuntimeCoreML

/// Tests for KV cache management components - Part 5 of the comprehensive test suite
@available(iOS 15.0, macOS 12.0, *)
final class KVCacheTests: XCTestCase {

    // MARK: - Helper Methods

    private func makeArray(shape: [Int], start: Float = 0) -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        let array = try! MLMultiArray(shape: nsShape, dataType: .float16)
        for i in 0..<array.count {
            setFloat(in: array, at: i, value: start + Float(i))
        }
        return array
    }

    private func setFloat(in array: MLMultiArray, at index: Int, value: Float) {
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        ptr[index] = Float16(value)
    }

    private func getFloat(from array: MLMultiArray, at index: Int) -> Float {
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        return Float(ptr[index])
    }

    // MARK: - LayerKVCache Tests

    func testCacheInitializationShapes() {
        let cache = LayerKVCache(maxSequenceLength: 4, numberOfHeads: 2, headDimension: 3)
        let key = cache.getCurrentKeyCache()
        let value = cache.getCurrentValueCache()

        XCTAssertEqual(key.shape.map { $0.intValue }, [2, 4, 3])
        XCTAssertEqual(value.shape.map { $0.intValue }, [2, 4, 3])

        // Verify arrays are zero-initialized
        for i in 0..<key.count {
            XCTAssertEqual(getFloat(from: key, at: i), 0)
            XCTAssertEqual(getFloat(from: value, at: i), 0)
        }
    }

    func testCacheUpdateAtSpecificPosition() {
        let cache = LayerKVCache(maxSequenceLength: 4, numberOfHeads: 2, headDimension: 3)
        let keyInput = makeArray(shape: [2, 3], start: 1)
        let valueInput = makeArray(shape: [2, 3], start: 5)

        cache.update(key: keyInput, value: valueInput, position: 0)
        let retrievedKeys = cache.getKeys(upToPosition: 0)
        let retrievedValues = cache.getValues(upToPosition: 0)

        XCTAssertEqual(retrievedKeys.shape.map { $0.intValue }, [2, 1, 3])
        XCTAssertEqual(retrievedValues.shape.map { $0.intValue }, [2, 1, 3])

        for i in 0..<keyInput.count {
            XCTAssertEqual(getFloat(from: retrievedKeys, at: i), getFloat(from: keyInput, at: i))
            XCTAssertEqual(getFloat(from: retrievedValues, at: i), getFloat(from: valueInput, at: i))
        }
    }

    func testCacheRetrievalUpToCurrentPosition() {
        let cache = LayerKVCache(maxSequenceLength: 4, numberOfHeads: 1, headDimension: 2)
        let firstKey = makeArray(shape: [1, 2], start: 1)
        let firstVal = makeArray(shape: [1, 2], start: 10)
        let secondKey = makeArray(shape: [1, 2], start: 3)
        let secondVal = makeArray(shape: [1, 2], start: 12)

        cache.update(key: firstKey, value: firstVal, position: 0)
        cache.update(key: secondKey, value: secondVal, position: 1)

        let keys = cache.getKeys(upToPosition: 1)
        let values = cache.getValues(upToPosition: 1)

        XCTAssertEqual(keys.shape.map { $0.intValue }, [1, 2, 2])
        XCTAssertEqual(values.shape.map { $0.intValue }, [1, 2, 2])

        // Verify first position
        XCTAssertEqual(getFloat(from: keys, at: 0), getFloat(from: firstKey, at: 0))
        XCTAssertEqual(getFloat(from: values, at: 0), getFloat(from: firstVal, at: 0))
        // Verify second position
        let secondIndex = firstKey.count
        XCTAssertEqual(getFloat(from: keys, at: secondIndex), getFloat(from: secondKey, at: 0))
        XCTAssertEqual(getFloat(from: values, at: secondIndex), getFloat(from: secondVal, at: 0))
    }

    func testCacheResetClearsData() {
        let cache = LayerKVCache(maxSequenceLength: 2, numberOfHeads: 1, headDimension: 2)
        let keyInput = makeArray(shape: [1, 2], start: 1)
        let valueInput = makeArray(shape: [1, 2], start: 3)

        cache.update(key: keyInput, value: valueInput, position: 0)
        cache.reset()

        let key = cache.getCurrentKeyCache()
        let value = cache.getCurrentValueCache()
        for i in 0..<key.count {
            XCTAssertEqual(getFloat(from: key, at: i), 0)
            XCTAssertEqual(getFloat(from: value, at: i), 0)
        }
    }

    func testCacheOverflowHandling() {
        let cache = LayerKVCache(maxSequenceLength: 2, numberOfHeads: 1, headDimension: 1)
        let valid = makeArray(shape: [1, 1], start: 1)
        cache.update(key: valid, value: valid, position: 0)
        let snapshot = cache.getCurrentKeyCache().copy() as! MLMultiArray

        // Attempt to write beyond max sequence length
        cache.update(key: valid, value: valid, position: 5)

        // Cache should remain unchanged
        let after = cache.getCurrentKeyCache()
        for i in 0..<after.count {
            XCTAssertEqual(getFloat(from: after, at: i), getFloat(from: snapshot, at: i))
        }
    }

    func testCacheSlicingOperations() {
        let cache = LayerKVCache(maxSequenceLength: 4, numberOfHeads: 1, headDimension: 1)
        let arr0 = makeArray(shape: [1, 1], start: 0)
        let arr1 = makeArray(shape: [1, 1], start: 1)
        let arr2 = makeArray(shape: [1, 1], start: 2)
        cache.update(key: arr0, value: arr0, position: 0)
        cache.update(key: arr1, value: arr1, position: 1)
        cache.update(key: arr2, value: arr2, position: 2)

        let slice = cache.getKeys(upToPosition: 1)
        XCTAssertEqual(slice.shape.map { $0.intValue }, [1, 2, 1])
        XCTAssertEqual(getFloat(from: slice, at: 0), getFloat(from: arr0, at: 0))
        XCTAssertEqual(getFloat(from: slice, at: 1), getFloat(from: arr1, at: 0))
    }

    func testConcurrentCacheUpdates() {
        let cache = LayerKVCache(maxSequenceLength: 4, numberOfHeads: 1, headDimension: 1)
        DispatchQueue.concurrentPerform(iterations: 4) { pos in
            let key = self.makeArray(shape: [1, 1], start: Float(pos))
            cache.update(key: key, value: key, position: pos)
        }
        let keys = cache.getKeys(upToPosition: 3)
        for pos in 0..<4 {
            XCTAssertEqual(getFloat(from: keys, at: pos), Float(pos))
        }
    }

    // MARK: - Multi-Layer Cache Tests

    func testCacheConcatenationAcrossLayers() throws {
        let config = TransformerConfig(
            vocabSize: 10,
            maxSequenceLength: 4,
            embeddingDimension: 8,
            numberOfHeads: 2,
            numberOfLayers: 2
        )
        let model = StatefulTransformerModel(config: config)
        let weights = try createMockWeights(config: config)
        let embedding = try createMockTokenEmbedding(embeddingDimension: config.embeddingDimension)

        let result = try model.forward(tokenEmbedding: embedding, weights: weights)
        let expectedFirstDim = config.numberOfLayers * config.numberOfHeads
        XCTAssertEqual(result.keyCache.shape.map { $0.intValue }, [expectedFirstDim, config.maxSequenceLength, config.headDimension])
        XCTAssertEqual(result.valueCache.shape.map { $0.intValue }, [expectedFirstDim, config.maxSequenceLength, config.headDimension])
    }

    func testCacheStateAfterModelReset() throws {
        let config = TransformerConfig(
            vocabSize: 10,
            maxSequenceLength: 4,
            embeddingDimension: 8,
            numberOfHeads: 1,
            numberOfLayers: 1
        )
        let model = StatefulTransformerModel(config: config)
        let weights = try createMockWeights(config: config)
        let embedding = try createMockTokenEmbedding(embeddingDimension: config.embeddingDimension)

        // Fill cache with two positions
        _ = try model.forward(tokenEmbedding: embedding, weights: weights)
        _ = try model.forward(tokenEmbedding: embedding, weights: weights)

        model.reset()
        let result = try model.forward(tokenEmbedding: embedding, weights: weights)

        // Position 1 should be zero after reset since only position 0 was written
        let keys = result.keyCache
        let index = 1 * config.headDimension
        XCTAssertEqual(getFloat(from: keys, at: index), 0)
    }

    func testIndependentLayerCacheManagement() {
        let cache1 = LayerKVCache(maxSequenceLength: 2, numberOfHeads: 1, headDimension: 1)
        let cache2 = LayerKVCache(maxSequenceLength: 2, numberOfHeads: 1, headDimension: 1)
        let arr = makeArray(shape: [1, 1], start: 5)

        cache1.update(key: arr, value: arr, position: 0)
        let key1 = cache1.getCurrentKeyCache()
        let key2 = cache2.getCurrentKeyCache()

        XCTAssertEqual(getFloat(from: key1, at: 0), 5)
        XCTAssertEqual(getFloat(from: key2, at: 0), 0)
    }

    // MARK: - Weight Helpers

    private func createMockWeights(config: TransformerConfig) throws -> TransformerWeights {
        let embeddings = try MLMultiArray(
            shape: [NSNumber(value: config.vocabSize), NSNumber(value: config.embeddingDimension)],
            dataType: .float16
        )
        var layers: [LayerWeights] = []
        for _ in 0..<config.numberOfLayers {
            let layer = LayerWeights(
                preAttentionNorm: createMockLayerNormWeights(dimension: config.embeddingDimension),
                attention: try createMockAttentionWeights(config: config),
                preMlpNorm: createMockLayerNormWeights(dimension: config.embeddingDimension),
                mlp: try createMockMlpWeights(config: config)
            )
            layers.append(layer)
        }
        let finalNorm = createMockLayerNormWeights(dimension: config.embeddingDimension)
        let outputProjection = LinearWeights(
            weight: try MLMultiArray(
                shape: [NSNumber(value: config.embeddingDimension), NSNumber(value: config.vocabSize)],
                dataType: .float16
            ),
            bias: nil
        )
        return TransformerWeights(
            embeddings: embeddings,
            layers: layers,
            finalNorm: finalNorm,
            outputProjection: outputProjection
        )
    }

    private func createMockLayerNormWeights(dimension: Int) -> LayerNormWeights {
        let weight = try! MLMultiArray(shape: [NSNumber(value: dimension)], dataType: .float16)
        return LayerNormWeights(weight: weight, bias: nil)
    }

    private func createMockAttentionWeights(config: TransformerConfig) throws -> AttentionWeights {
        let embeddingDim = config.embeddingDimension
        return AttentionWeights(
            queryProjection: LinearWeights(
                weight: try MLMultiArray(shape: [NSNumber(value: embeddingDim), NSNumber(value: embeddingDim)], dataType: .float16),
                bias: nil
            ),
            keyProjection: LinearWeights(
                weight: try MLMultiArray(shape: [NSNumber(value: embeddingDim), NSNumber(value: embeddingDim)], dataType: .float16),
                bias: nil
            ),
            valueProjection: LinearWeights(
                weight: try MLMultiArray(shape: [NSNumber(value: embeddingDim), NSNumber(value: embeddingDim)], dataType: .float16),
                bias: nil
            ),
            outputProjection: LinearWeights(
                weight: try MLMultiArray(shape: [NSNumber(value: embeddingDim), NSNumber(value: embeddingDim)], dataType: .float16),
                bias: nil
            )
        )
    }

    private func createMockMlpWeights(config: TransformerConfig) throws -> MlpWeights {
        let embeddingDim = config.embeddingDimension
        let mlpDim = embeddingDim * 4
        return MlpWeights(
            gateProjection: LinearWeights(
                weight: try MLMultiArray(shape: [NSNumber(value: embeddingDim), NSNumber(value: mlpDim)], dataType: .float16),
                bias: nil
            ),
            upProjection: LinearWeights(
                weight: try MLMultiArray(shape: [NSNumber(value: embeddingDim), NSNumber(value: mlpDim)], dataType: .float16),
                bias: nil
            ),
            downProjection: LinearWeights(
                weight: try MLMultiArray(shape: [NSNumber(value: mlpDim), NSNumber(value: embeddingDim)], dataType: .float16),
                bias: nil
            )
        )
    }

    private func createMockTokenEmbedding(embeddingDimension: Int) throws -> MLMultiArray {
        try MLMultiArray(shape: [NSNumber(value: 1), NSNumber(value: embeddingDimension)], dataType: .float16)
    }
}

#endif
