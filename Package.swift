// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "Lilims",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "Lilims", targets: ["Lilims"]),
        .executable(name: "LilimsApp", targets: ["LilimsApp"])
    ],
    targets: [
        .target(
            name: "RuntimeCoreML",
            path: "Sources/Runtime/CoreML"
        ),
        .target(
            name: "Lilims",
            dependencies: ["RuntimeCoreML"],
            path: "Sources/Lilims"
        ),
        .executableTarget(
            name: "LilimsApp",
            dependencies: ["Lilims"],
            path: "Sources/UI",
            exclude: ["Info.plist"]
        ),
        .testTarget(
            name: "LilimsTests",
            dependencies: ["Lilims"],
            path: "Tests/LilimsTests"
        )
    ]
)
