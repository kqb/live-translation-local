// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OmiApp",
    platforms: [
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "OmiApp",
            targets: ["OmiApp"]
        )
    ],
    dependencies: [],
    targets: [
        .target(
            name: "OmiApp",
            dependencies: [],
            path: "OmiApp"
        )
    ]
)
