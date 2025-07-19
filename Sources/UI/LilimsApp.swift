#if canImport(SwiftUI)
import SwiftUI
import Lilims

@main
struct LilimsApp: App {
    var body: some Scene {
        WindowGroup {
            Text(Lilims().greet())
        }
    }
}
#else
import Lilims
import Foundation
@main
enum LilimsApp {
    static func main() {
        print(Lilims().greet())
    }
}
#endif
