// The Core ML backend is only built on platforms where CoreML is available.
#if canImport(CoreML)
import CoreML
import Foundation

/// Backend that runs language models using Core ML stateful evaluation.
public final class CoreMLBackend {
    private let model: MLModel
    private var state: MLFeatureProvider?
    private var kvCache: KVCache?
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
    }

    /// Generates up to `maxTokens` continuations for the given `prompt`.
    /// - Returns: The combined prompt and generated tokens.
    public func generate(prompt: [Int32], maxTokens: Int) throws -> [Int32] {
        var tokens = prompt
        let options = MLPredictionOptions()
        options.usesCPUOnly = false
        for _ in 0..<maxTokens {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "token": MLFeatureValue(int32: tokens.last ?? 0)
            ])
            let combinedInput: MLFeatureProvider
            if let state = state {
                combinedInput = MLDictionaryFeatureProvider(from: input, merging: state)
            } else {
                combinedInput = input
            }
            let output = try model.prediction(from: combinedInput, options: options)
            state = output
            updateCache(from: output)
            guard let next = output.featureValue(for: "token")?.int32Value else {
                break
            }
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
    public func stream(prompt: [Int32], maxTokens: Int) -> AsyncThrowingStream<Int32, Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            Task {
                do {
                    var tokens = prompt
                    let options = MLPredictionOptions()
                    options.usesCPUOnly = false
                    for _ in 0..<maxTokens {
                        let input = try MLDictionaryFeatureProvider(dictionary: [
                            "token": MLFeatureValue(int32: tokens.last ?? 0)
                        ])
                        let combinedInput: MLFeatureProvider
                        if let state = state {
                            combinedInput = try MLDictionaryFeatureProvider(from: input, merging: state)
                        } else {
                            combinedInput = input
                        }
                        let output = try model.prediction(from: combinedInput, options: options)
                        state = output
                        updateCache(from: output)
                        guard let next = output.featureValue(for: "token")?.int32Value else {
                            break
                        }
                        tokens.append(next)
                        delegate?.didGenerate(token: next)
                        while continuation.yield(next) == .dropped {
                            await Task.yield()
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Extracts key/value tensors from *output* and stores them in the cache.
    private func updateCache(from output: MLFeatureProvider) {
        guard let keyArray = output.featureValue(for: "key")?.multiArrayValue,
              let valueArray = output.featureValue(for: "value")?.multiArrayValue else {
            return
        }
        let key = MLShapedArray<Float16>(keyArray)
        let value = MLShapedArray<Float16>(valueArray)
        kvCache?.append(key: key, value: value)
    }
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
    private var keys: [MLShapedArray<Float16>] = []
    private var values: [MLShapedArray<Float16>] = []

    /// Creates a cache with room for `capacity` tokens.
    /// - Parameter capacity: Maximum number of tokens to keep.
    init(capacity: Int) {
        self.capacity = capacity
    }

    /// Appends a key/value pair and evicts the least-recently used entry if needed.
    /// - Parameters:
    ///   - key: Key tensor for the token.
    ///   - value: Value tensor for the token.
    func append(key: MLShapedArray<Float16>, value: MLShapedArray<Float16>) {
        keys.append(key)
        values.append(value)
        if keys.count > capacity {
            keys.removeFirst()
            values.removeFirst()
        }
    }
}
#endif
