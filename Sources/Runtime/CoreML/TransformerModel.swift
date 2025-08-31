#if canImport(CoreML)
import CoreML
import Foundation

/// Configuration for transformer model architecture.
public struct TransformerConfig {
    public let vocabSize: Int
    public let maxSequenceLength: Int
    public let embeddingDimension: Int
    public let numberOfHeads: Int
    public let numberOfLayers: Int
    public let ropeBase: Float
    
    public var headDimension: Int {
        embeddingDimension / numberOfHeads
    }
    
    public init(
        vocabSize: Int,
        maxSequenceLength: Int,
        embeddingDimension: Int,
        numberOfHeads: Int,
        numberOfLayers: Int,
        ropeBase: Float = 10_000
    ) {
        self.vocabSize = vocabSize
        self.maxSequenceLength = maxSequenceLength
        self.embeddingDimension = embeddingDimension
        self.numberOfHeads = numberOfHeads
        self.numberOfLayers = numberOfLayers
        self.ropeBase = ropeBase
    }
}

/// Stateful transformer model for autoregressive generation using CoreML.
@available(iOS 15.0, macOS 12.0, *)
public final class StatefulTransformerModel {
    private let config: TransformerConfig
    private let ropeFreqs: (sine: MLShapedArray<Float32>, cosine: MLShapedArray<Float32>)
    private var kvCache: [LayerKVCache]
    private var currentPosition: Int = 0
    
    public init(config: TransformerConfig) {
        self.config = config
        self.ropeFreqs = Rope.rotaryTables(
            sequenceLength: config.maxSequenceLength,
            headDimension: config.headDimension,
            base: config.ropeBase
        )
        self.kvCache = (0..<config.numberOfLayers).map { _ in
            LayerKVCache(
                maxSequenceLength: config.maxSequenceLength,
                numberOfHeads: config.numberOfHeads,
                headDimension: config.headDimension
            )
        }
    }
    
    /// Reset the model state for new generation.
    public func reset() {
        currentPosition = 0
        kvCache.forEach { $0.reset() }
    }
    
    /// Process a single token and return logits with updated KV cache.
    public func forward(
        tokenEmbedding: MLMultiArray,
        weights: TransformerWeights
    ) throws -> (logits: MLMultiArray, keyCache: MLMultiArray, valueCache: MLMultiArray) {
        var hidden = tokenEmbedding
        
        // Process through transformer layers
        for layerIndex in 0..<config.numberOfLayers {
            hidden = try transformerLayer(
                input: hidden,
                layerWeights: weights.layers[layerIndex],
                layerIndex: layerIndex,
                position: currentPosition
            )
        }
        
        // Apply final layer norm and projection to logits
        let normalized = try MLArrayUtils.layerNorm(hidden, weights: weights.finalNorm)
        let logits = try MLArrayUtils.linear(normalized, weights: weights.outputProjection)
        
        currentPosition += 1
        
        // Concatenate all layer caches for output
        let keyCache = try concatenateKVCache(type: .key)
        let valueCache = try concatenateKVCache(type: .value)
        
        return (logits: logits, keyCache: keyCache, valueCache: valueCache)
    }
    
    private func transformerLayer(
        input: MLMultiArray,
        layerWeights: LayerWeights,
        layerIndex: Int,
        position: Int
    ) throws -> MLMultiArray {
        // Pre-attention layer norm
        let normalizedInput = try MLArrayUtils.layerNorm(input, weights: layerWeights.preAttentionNorm)
        
        // Multi-head attention with KV caching
        let attentionOutput = try multiHeadAttention(
            input: normalizedInput,
            weights: layerWeights.attention,
            layerIndex: layerIndex,
            position: position
        )
        
        // Residual connection
        let afterAttention = try MLArrayUtils.addTensors(input, attentionOutput)
        
        // Pre-MLP layer norm
        let normalizedAfterAttention = try MLArrayUtils.layerNorm(afterAttention, weights: layerWeights.preMlpNorm)
        
        // MLP
        let mlpOutput = try mlpBlock(normalizedAfterAttention, weights: layerWeights.mlp)
        
        // Final residual connection
        return try MLArrayUtils.addTensors(afterAttention, mlpOutput)
    }
    
    private func multiHeadAttention(
        input: MLMultiArray,
        weights: AttentionWeights,
        layerIndex: Int,
        position: Int
    ) throws -> MLMultiArray {
        // Project to Q, K, V
        let queries = try MLArrayUtils.linear(input, weights: weights.queryProjection)
        let keys = try MLArrayUtils.linear(input, weights: weights.keyProjection) 
        let values = try MLArrayUtils.linear(input, weights: weights.valueProjection)
        
        // Reshape for multi-head attention
        let queryHeads = try MLArrayUtils.reshapeForHeads(queries, numberOfHeads: config.numberOfHeads)
        let keyHeads = try MLArrayUtils.reshapeForHeads(keys, numberOfHeads: config.numberOfHeads)
        let valueHeads = try MLArrayUtils.reshapeForHeads(values, numberOfHeads: config.numberOfHeads)
        
        // Apply rotary position embeddings
        let (rotatedQueries, rotatedKeys) = try applyRotaryEmbeddings(
            queries: queryHeads,
            keys: keyHeads,
            position: position
        )
        
        // Update KV cache
        kvCache[layerIndex].update(key: rotatedKeys, value: valueHeads, position: position)
        
        // Get cached keys and values for attention
        let cachedKeys = kvCache[layerIndex].getKeys(upToPosition: position)
        let cachedValues = kvCache[layerIndex].getValues(upToPosition: position)
        
        // Compute attention
        let attentionOutput = try computeAttention(
            queries: rotatedQueries,
            keys: cachedKeys,
            values: cachedValues
        )
        
        // Project back to embedding dimension
        let reshapedOutput = try MLArrayUtils.reshapeFromHeads(attentionOutput)
        return try MLArrayUtils.linear(reshapedOutput, weights: weights.outputProjection)
    }
    
    private func applyRotaryEmbeddings(
        queries: MLMultiArray,
        keys: MLMultiArray,
        position: Int
    ) throws -> (MLMultiArray, MLMultiArray) {
        let halfDim = config.headDimension / 2
        
        // Get rotation matrices for current position
        var sinValues = [Float32]()
        var cosValues = [Float32]()
        for i in 0..<halfDim {
            sinValues.append(ropeFreqs.sine[position, i].scalar ?? 0)
            cosValues.append(ropeFreqs.cosine[position, i].scalar ?? 1)
        }
        
        // Apply rotation to queries and keys
        let rotatedQueries = try Rope.rotateHalf(queries, sin: sinValues, cos: cosValues, numberOfHeads: config.numberOfHeads)
        let rotatedKeys = try Rope.rotateHalf(keys, sin: sinValues, cos: cosValues, numberOfHeads: config.numberOfHeads)
        
        return (rotatedQueries, rotatedKeys)
    }
    
    private func computeAttention(
        queries: MLMultiArray,
        keys: MLMultiArray,
        values: MLMultiArray
    ) throws -> MLMultiArray {
        let scale = 1.0 / sqrt(Float(config.headDimension))
        
        // Compute attention scores: Q * K^T
        let scores = try MLArrayUtils.matrixMultiply(queries, MLArrayUtils.transpose(keys))
        
        // Scale scores
        let scaledScores = try MLArrayUtils.scalarMultiply(scores, scale)
        
        // Apply causal mask (prevent attention to future tokens)
        let maskedScores = try MLArrayUtils.applyCausalMask(scaledScores, currentLength: currentPosition + 1)
        
        // Softmax
        let attentionWeights = try MLArrayUtils.softmax(maskedScores)
        
        // Apply attention to values: Attention * V
        return try MLArrayUtils.matrixMultiply(attentionWeights, values)
    }
    
    private func mlpBlock(_ input: MLMultiArray, weights: MlpWeights) throws -> MLMultiArray {
        // Two-layer MLP with activation
        let gateOutput = try MLArrayUtils.linear(input, weights: weights.gateProjection)
        let upOutput = try MLArrayUtils.linear(input, weights: weights.upProjection)
        let activated = try MLArrayUtils.silu(gateOutput)
        let combined = try MLArrayUtils.multiplyTensors(activated, upOutput)
        return try MLArrayUtils.linear(combined, weights: weights.downProjection)
    }
    
    private enum CacheType {
        case key, value
    }
    
    private func concatenateKVCache(type: CacheType) throws -> MLMultiArray {
        // Concatenate caches from all layers for output
        let cacheArrays = kvCache.map { cache in
            switch type {
            case .key: return cache.getCurrentKeyCache()
            case .value: return cache.getCurrentValueCache()
            }
        }
        
        return try MLArrayUtils.concatenateArrays(cacheArrays, axis: 0)
    }
}

/// KV cache for a single transformer layer.
private final class LayerKVCache {
    private let maxSequenceLength: Int
    private let numberOfHeads: Int
    private let headDimension: Int
    private var keyCache: MLMultiArray
    private var valueCache: MLMultiArray
    private var currentLength: Int = 0
    
    init(maxSequenceLength: Int, numberOfHeads: Int, headDimension: Int) {
        self.maxSequenceLength = maxSequenceLength
        self.numberOfHeads = numberOfHeads
        self.headDimension = headDimension
        
        let shape = [numberOfHeads, maxSequenceLength, headDimension].map { NSNumber(value: $0) }
        self.keyCache = try! MLMultiArray(shape: shape, dataType: .float16)
        self.valueCache = try! MLMultiArray(shape: shape, dataType: .float16)
    }
    
    func reset() {
        currentLength = 0
        // Clear cache arrays
        let zeroData = Data(repeating: 0, count: keyCache.count * 2) // float16 = 2 bytes
        keyCache.dataPointer.copyMemory(from: zeroData.withUnsafeBytes { $0.bindMemory(to: UInt8.self).baseAddress! }, byteCount: zeroData.count)
        valueCache.dataPointer.copyMemory(from: zeroData.withUnsafeBytes { $0.bindMemory(to: UInt8.self).baseAddress! }, byteCount: zeroData.count)
    }
    
    func update(key: MLMultiArray, value: MLMultiArray, position: Int) {
        // Store key and value at the current position
        for headIdx in 0..<numberOfHeads {
            for dimIdx in 0..<headDimension {
                let cacheKeyIndex = [NSNumber(value: headIdx), NSNumber(value: position), NSNumber(value: dimIdx)]
                let cacheValueIndex = [NSNumber(value: headIdx), NSNumber(value: position), NSNumber(value: dimIdx)]
                let inputIndex = [NSNumber(value: headIdx), NSNumber(value: dimIdx)]
                
                keyCache[cacheKeyIndex] = key[inputIndex]
                valueCache[cacheValueIndex] = value[inputIndex]
            }
        }
        currentLength = max(currentLength, position + 1)
    }
    
    func getKeys(upToPosition: Int) -> MLMultiArray {
        return getSlice(from: keyCache, length: upToPosition + 1)
    }
    
    func getValues(upToPosition: Int) -> MLMultiArray {
        return getSlice(from: valueCache, length: upToPosition + 1)
    }
    
    func getCurrentKeyCache() -> MLMultiArray {
        return keyCache
    }
    
    func getCurrentValueCache() -> MLMultiArray {
        return valueCache
    }
    
    private func getSlice(from cache: MLMultiArray, length: Int) -> MLMultiArray {
        let shape = [numberOfHeads, length, headDimension].map { NSNumber(value: $0) }
        let result = try! MLMultiArray(shape: shape, dataType: cache.dataType)
        
        for headIdx in 0..<numberOfHeads {
            for seqIdx in 0..<length {
                for dimIdx in 0..<headDimension {
                    let sourceIndex = [NSNumber(value: headIdx), NSNumber(value: seqIdx), NSNumber(value: dimIdx)]
                    let targetIndex = [NSNumber(value: headIdx), NSNumber(value: seqIdx), NSNumber(value: dimIdx)]
                    result[targetIndex] = cache[sourceIndex]
                }
            }
        }
        
        return result
    }
}

/// Weight tensors for the transformer model.
public struct TransformerWeights {
    public let embeddings: MLMultiArray
    public let layers: [LayerWeights]
    public let finalNorm: LayerNormWeights
    public let outputProjection: LinearWeights
}

public struct LayerWeights {
    public let preAttentionNorm: LayerNormWeights
    public let attention: AttentionWeights
    public let preMlpNorm: LayerNormWeights
    public let mlp: MlpWeights
}

public struct AttentionWeights {
    public let queryProjection: LinearWeights
    public let keyProjection: LinearWeights
    public let valueProjection: LinearWeights
    public let outputProjection: LinearWeights
}

public struct MlpWeights {
    public let gateProjection: LinearWeights
    public let upProjection: LinearWeights
    public let downProjection: LinearWeights
}

public struct LinearWeights {
    public let weight: MLMultiArray
    public let bias: MLMultiArray?
}

public struct LayerNormWeights {
    public let weight: MLMultiArray
    public let bias: MLMultiArray?
}

#endif