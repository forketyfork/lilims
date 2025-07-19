import XCTest
@testable import Lilims

final class LilimsTests: XCTestCase {
    func testGreet() {
        XCTAssertEqual(Lilims().greet(), "Hello")
    }
}
