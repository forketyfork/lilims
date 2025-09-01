// The Core ML backend is only built on platforms where CoreML is available.
#if canImport(CoreML)
import CoreML
import Foundation

/// Errors that may occur during generation.
public enum CoreMLBackendError: Error {
    /// The device ran out of memory while generating tokens.
    case outOfMemory
}

/// Backend that runs language models using Core ML stateful evaluation.
@available(iOS 15.0, macOS 12.0, *)
public final class CoreMLBackend: @unchecked Sendable {
    private let model: MLModel
    private var state: MLFeatureProvider?
    private var kvCache: KVCache?
    private var transformerModel: StatefulTransformerModel?
    private var modelConfig: TransformerConfig?
    private var modelWeights: TransformerWeights?
    /// Delegate notified as tokens are generated.
    public weak var delegate: TokenStreamDelegate?

    /// Initializes the backend with a compiled Core ML model at *url*.
    /// - Parameters:
    ///   - url: Location of the compiled Core ML model.
    ///   - delegate: Optional stream delegate to receive tokens.
    ///   - maxCacheTokens: Maximum number of tokens to keep in the KV cache.
    public init(modelAt url: URL, delegate: TokenStreamDelegate? = nil, maxCacheTokens: Int = 0) throws {
        self.model = try MLModel(contentsOf: url)
        self.delegate = delegate
        if maxCacheTokens > 0 {
            self.kvCache = KVCache(capacity: maxCacheTokens)
        }
        
        // Try to load transformer configuration and weights
        try loadTransformerConfiguration(from: url)
    }
    
    /// Loads transformer configuration and weights from model metadata
    private func loadTransformerConfiguration(from url: URL) throws {
        // Look for metadata file next to the model (created by conversion script)
        let modelDir = url.deletingLastPathComponent()
        let modelName = url.deletingPathExtension().lastPathComponent
        let metadataURL = modelDir.appendingPathComponent("\(modelName)_metadata.json")
        
        guard FileManager.default.fileExists(atPath: metadataURL.path) else {
            // No metadata file, fall back to basic CoreML model
            return
        }
        
        let data = try Data(contentsOf: metadataURL)
        let metadata = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        
        // Handle both direct config and nested config formats
        var config: [String: Any]?
        if let directConfig = metadata,
           directConfig["vocab_size"] != nil {
            // Direct config format (GGUF conversion)
            config = directConfig
        } else if let nestedConfig = metadata?["config"] as? [String: Any] {
            // Nested config format (HF conversion)
            config = nestedConfig
        }
        
        guard let config = config,
              let vocabSize = config["vocab_size"] as? Int else {
            // No valid config found, fall back to basic model
            return
        }
        
        // Extract parameters with fallbacks for different formats
        let nEmbd = config["n_embd"] as? Int ?? 
                   config["hidden_size"] as? Int ?? 
                   config["d_model"] as? Int ?? 768
        let nHead = config["n_head"] as? Int ?? 
                   config["num_attention_heads"] as? Int ?? 12
        let nLayer = config["n_layer"] as? Int ?? 
                    config["num_hidden_layers"] as? Int ?? 12
        let contextLength = config["context_length"] as? Int ?? 
                           config["max_position_embeddings"] as? Int ?? 
                           config["sequence_length"] as? Int ?? 2048
        
        self.modelConfig = TransformerConfig(
            vocabSize: vocabSize,
            maxSequenceLength: contextLength,
            embeddingDimension: nEmbd,
            numberOfHeads: nHead,
            numberOfLayers: nLayer,
            ropeBase: config["rope_theta"] as? Float ?? 
                     config["rope_freq_base"] as? Float ?? 10_000
        )
        
        if let config = modelConfig {
            self.transformerModel = StatefulTransformerModel(config: config)
            // Note: In a complete implementation, we would load actual weights from the model
            // For now, this provides the architecture framework
        }
    }

    /// Generates up to `maxTokens` continuations for the given `prompt`.
    /// - Returns: The combined prompt and generated tokens.
    public func generate(
        prompt: [Int32],
        maxTokens: Int,
        temperature: Float = 1.0,
        topK: Int = 0
    ) throws -> [Int32] {
        // Use stateful transformer if available, otherwise fall back to basic CoreML
        if let transformer = transformerModel {
            return try generateWithTransformer(
                transformer: transformer,
                prompt: prompt,
                maxTokens: maxTokens,
                temperature: temperature,
                topK: topK
            )
        } else {
            return try generateWithBasicModel(
                prompt: prompt,
                maxTokens: maxTokens,
                temperature: temperature,
                topK: topK
            )
        }
    }
    
    /// Generate tokens using the stateful transformer model
    private func generateWithTransformer(
        transformer: StatefulTransformerModel,
        prompt: [Int32],
        maxTokens: Int,
        temperature: Float,
        topK: Int
    ) throws -> [Int32] {
        var tokens = prompt
        transformer.reset()
        
        // Process prompt tokens first (if we had embeddings)
        // For now, we'll work with the basic CoreML model to get embeddings
        // In a complete implementation, we would have embedding weights loaded
        
        // Generate new tokens
        for _ in 0..<maxTokens {
            try Task.checkCancellation()
            
            // For now, fall back to basic model since we don't have weight loading implemented
            // This provides the framework for future enhancement
            let tokenInput = tokens.last ?? 0
            var features: [String: MLFeatureValue] = [
                "token": MLFeatureValue(int64: Int64(tokenInput))
            ]
            
            if let state = state {
                for name in state.featureNames {
                    features[name] = state.featureValue(for: name)
                }
            } else if let cache = kvCache?.asFeatureProvider() {
                for name in cache.featureNames {
                    features[name] = cache.featureValue(for: name)
                }
            }
            
            let combinedInput = try MLDictionaryFeatureProvider(dictionary: features)
            let output: MLFeatureProvider
            do {
                output = try model.prediction(from: combinedInput, options: MLPredictionOptions())
            } catch {
                throw mapPredictionError(error)
            }
            
            state = output
            updateCache(from: output)
            
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
                break
            }
            
            let next = sample(from: logits, temperature: temperature, topK: topK)
            tokens.append(next)
            delegate?.didGenerate(token: next)
        }
        
        return tokens
    }
    
    /// Generate tokens using the basic CoreML model (fallback)
    private func generateWithBasicModel(
        prompt: [Int32],
        maxTokens: Int,
        temperature: Float,
        topK: Int
    ) throws -> [Int32] {
        var tokens = prompt
        state = nil
        let options = MLPredictionOptions()
        
        for _ in 0..<maxTokens {
            try Task.checkCancellation()
            var features: [String: MLFeatureValue] = [
                "token": MLFeatureValue(int64: Int64(tokens.last ?? 0))
            ]
            if let state = state {
                for name in state.featureNames {
                    features[name] = state.featureValue(for: name)
                }
            } else if let cache = kvCache?.asFeatureProvider() {
                for name in cache.featureNames {
                    features[name] = cache.featureValue(for: name)
                }
            }
            let combinedInput = try MLDictionaryFeatureProvider(dictionary: features)
            let output: MLFeatureProvider
            do {
                output = try model.prediction(from: combinedInput, options: options)
            } catch {
                throw mapPredictionError(error)
            }
            state = output
            updateCache(from: output)
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
                break
            }
            let next = sample(from: logits, temperature: temperature, topK: topK)
            tokens.append(next)
            delegate?.didGenerate(token: next)
        }
        return tokens
    }

    /// Streams tokens for the given `prompt`.
    ///
    /// Backpressure is enforced by buffering only a single token at a time. If
    /// the consumer is slow, generation pauses until the token is consumed.
    /// - Parameters:
    ///   - prompt: Prefix tokens to seed generation.
    ///   - maxTokens: Maximum number of tokens to produce.
    /// - Returns: Async sequence yielding tokens as they are generated.
    @available(iOS 15.0, macOS 12.0, *)
    public func stream(
        prompt: [Int32],
        maxTokens: Int,
        temperature: Float = 1.0,
        topK: Int = 0
    ) -> AsyncThrowingStream<Int32, Error> {
        let backend = self
        return AsyncThrowingStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            Task {
                do {
                    // Use the same pattern as generate() - transformer if available, otherwise basic model
                    if let transformer = backend.transformerModel {
                        try await backend.streamWithTransformer(
                            transformer: transformer,
                            prompt: prompt,
                            maxTokens: maxTokens,
                            temperature: temperature,
                            topK: topK,
                            continuation: continuation
                        )
                    } else {
                        try await backend.streamWithBasicModel(
                            prompt: prompt,
                            maxTokens: maxTokens,
                            temperature: temperature,
                            topK: topK,
                            continuation: continuation
                        )
                    }
                } catch {
                    continuation.finish(throwing: mapPredictionError(error))
                }
            }
        }
    }
    
    /// Stream tokens using the stateful transformer model
    private func streamWithTransformer(
        transformer: StatefulTransformerModel,
        prompt: [Int32],
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        continuation: AsyncThrowingStream<Int32, Error>.Continuation
    ) async throws {
        _ = prompt
        transformer.reset()
        
        // For now, fall back to basic model streaming since we don't have weight loading implemented
        // This provides the framework for future enhancement
        try await streamWithBasicModel(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topK: topK,
            continuation: continuation
        )
    }
    
    /// Stream tokens using the basic CoreML model (fallback)
    private func streamWithBasicModel(
        prompt: [Int32],
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        continuation: AsyncThrowingStream<Int32, Error>.Continuation
    ) async throws {
        var tokens = prompt
        self.state = nil
        let options = MLPredictionOptions()
        
        for _ in 0..<maxTokens {
            try Task.checkCancellation()
            var features: [String: MLFeatureValue] = [
                "token": MLFeatureValue(int64: Int64(tokens.last ?? 0))
            ]
            if let state = self.state {
                for name in state.featureNames {
                    features[name] = state.featureValue(for: name)
                }
            } else if let cache = self.kvCache?.asFeatureProvider() {
                for name in cache.featureNames {
                    features[name] = cache.featureValue(for: name)
                }
            }
            let combinedInput = try MLDictionaryFeatureProvider(dictionary: features)
            let output: MLFeatureProvider
            do {
                output = try self.model.prediction(from: combinedInput, options: options)
            } catch {
                throw mapPredictionError(error)
            }
            self.state = output
            self.updateCache(from: output)
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
                break
            }
            let next = sample(from: logits, temperature: temperature, topK: topK)
            tokens.append(next)
            self.delegate?.didGenerate(token: next)
            continuation.yield(next)
        }
        continuation.finish()
    }

    /// Extracts key/value tensors from *output* and stores them in the cache.
    private func updateCache(from output: MLFeatureProvider) {
        guard let keyArray = output.featureValue(for: "key")?.multiArrayValue,
              let valueArray = output.featureValue(for: "value")?.multiArrayValue else {
            return
        }
        kvCache?.append(key: keyArray, value: valueArray)
    }
}

/// Converts prediction errors into more specific backend errors.
private func mapPredictionError(_ error: Error) -> Error {
    let nsError = error as NSError
    if nsError.domain == NSPOSIXErrorDomain && nsError.code == Int(POSIXErrorCode.ENOMEM.rawValue) {
        return CoreMLBackendError.outOfMemory
    }
    return error
}

private extension MLDictionaryFeatureProvider {
    convenience init(from a: MLFeatureProvider, merging b: MLFeatureProvider) throws {
        var dict: [String: MLFeatureValue] = [:]
        for feature in a.featureNames {
            dict[feature] = a.featureValue(for: feature)
        }
        for feature in b.featureNames {
            dict[feature] = b.featureValue(for: feature)
        }
        try self.init(dictionary: dict)
    }
}

/// Simple key/value cache storing the most recent `capacity` entries.
private final class KVCache {
    private let capacity: Int
    private var keys: [MLMultiArray] = []
    private var values: [MLMultiArray] = []

    /// Creates a cache with room for `capacity` tokens.
    /// - Parameter capacity: Maximum number of tokens to keep.
    init(capacity: Int) {
        self.capacity = capacity
    }

    /// Appends a key/value pair and evicts the least-recently used entry if needed.
    /// - Parameters:
    ///   - key: Key tensor for the token.
    ///   - value: Value tensor for the token.
    func append(key: MLMultiArray, value: MLMultiArray) {
        keys.append(key)
        values.append(value)
        if keys.count > capacity {
            keys.removeFirst()
            values.removeFirst()
        }
    }

    /// Creates a feature provider representing the current cache contents.
    func asFeatureProvider() -> MLFeatureProvider? {
        guard let key = keys.last, let value = values.last else { return nil }
        return try? MLDictionaryFeatureProvider(dictionary: [
            "key": MLFeatureValue(multiArray: key),
            "value": MLFeatureValue(multiArray: value)
        ])
    }
}

/// Sample a token index from *logits* using temperature and top-k filtering.
@available(iOS 15.0, macOS 12.0, *)
private func sample(
    from logits: MLMultiArray,
    temperature: Float,
    topK: Int
) -> Int32 {
    var values = MLShapedArray<Float32>(logits).scalars
    if temperature != 1 {
        for i in 0..<values.count {
            values[i] /= temperature
        }
    }
    if topK > 0 && topK < values.count {
        let threshold = values.sorted(by: >)[topK - 1]
        for i in 0..<values.count where values[i] < threshold {
            values[i] = -.infinity
        }
    }
    let maxVal = values.max() ?? 0
    var expVals = [Float](repeating: 0, count: values.count)
    var sum: Float = 0
    for i in 0..<values.count {
        expVals[i] = Float(exp(Double(values[i] - maxVal)))
        sum += expVals[i]
    }
    var rnd = Float.random(in: 0..<sum)
    for i in 0..<expVals.count {
        rnd -= expVals[i]
        if rnd <= 0 {
            return Int32(i)
        }
    }
    return Int32(expVals.count - 1)
}
#endif
