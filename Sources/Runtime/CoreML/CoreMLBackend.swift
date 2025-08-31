// The Core ML backend is only built on platforms where CoreML is available.
#if canImport(CoreML)
import CoreML
import Foundation

/// Backend that runs language models using Core ML stateful evaluation.
public final class CoreMLBackend {
    private let model: MLModel
    private var state: MLFeatureProvider?
    /// Delegate notified as tokens are generated.
    public weak var delegate: TokenStreamDelegate?

    /// Initializes the backend with a compiled Core ML model at *url*.
    /// - Parameters:
    ///   - url: Location of the compiled Core ML model.
    ///   - delegate: Optional stream delegate to receive tokens.
    public init(modelAt url: URL, delegate: TokenStreamDelegate? = nil) throws {
        self.model = try MLModel(contentsOf: url)
        self.delegate = delegate
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
            guard let next = output.featureValue(for: "token")?.int32Value else {
                break
            }
            tokens.append(next)
            delegate?.didGenerate(token: next)
        }
        return tokens
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
#endif
