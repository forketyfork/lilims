#if canImport(CoreML)
import XCTest
@testable import RuntimeCoreML

final class RuntimeCoreMLTests: XCTestCase {
    func testDecodeTinyStories() throws {
        // Skip this test on CI since the model file won't be available
        if ProcessInfo.processInfo.environment["CI"] != nil {
            throw XCTSkip("TinyStories model not available on CI")
        }
        
        guard let url = Bundle(for: type(of: self)).url(forResource: "tinystories", withExtension: "mlpackage") else {
            throw XCTSkip("TinyStories model not available")
        }
        let backend = try CoreMLBackend(modelAt: url, maxCacheTokens: 32)
        let prompt: [Int32] = [1]
        let tokens = try backend.generate(prompt: prompt, maxTokens: 20)
        XCTAssertEqual(tokens.count, prompt.count + 20)
    }
}
#endif
