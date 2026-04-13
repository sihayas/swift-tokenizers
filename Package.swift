// swift-tools-version: 6.1

import PackageDescription

let tokenizerCoreSources = [
    "BinaryDistinct.swift",
    "Config.swift",
    "Tokenizer.swift",
    "TokenizerCompatibility.swift",
    "TokenizerRuntimeConfiguration.swift",
]

let tokenizerSwiftBackendSources = [
    "BPETokenizer.swift",
    "BertTokenizer.swift",
    "ByteEncoder.swift",
    "Decoder.swift",
    "Normalizer.swift",
    "PostProcessor.swift",
    "PreTokenizer.swift",
    "String+PreTokenization.swift",
    "TokenLattice.swift",
    "Trie.swift",
    "UnigramTokenizer.swift",
    "YYJSONParser.swift",
    "SwiftTokenizerBackend.swift",
]

let tokenizerRustBackendSources = [
    "RustBackedTokenizer.swift"
]

let tokenizerDirectorySources =
    tokenizerCoreSources
    + tokenizerSwiftBackendSources
    + tokenizerRustBackendSources

let benchmarksEnabled = Context.environment["TOKENIZERS_ENABLE_BENCHMARKS"] == "1"
let localRustArtifactPath = Context.environment["TOKENIZERS_RUST_LOCAL_XCFRAMEWORK_PATH"]

func excludedTokenizerSources(keeping sources: [String]) -> [String] {
    tokenizerDirectorySources.filter { !sources.contains($0) }
}

var packageDependencies: [Package.Dependency] = [
    .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0"),
    .package(url: "https://github.com/ibireme/yyjson.git", exact: "0.12.0"),
    .package(url: "https://github.com/DePasqualeOrg/swift-hf-api.git", from: "0.2.0"),
]

if benchmarksEnabled {
    packageDependencies.append(
        .package(
            // TODO: Switch to a major version pin once mlx-swift-lm publishes a new major release that includes these APIs.
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            revision: "f7c5c99e54112845242b7f46d1d6335fcbe57476"
        )
    )
}

let tokenizersRustTarget: Target =
    if let localRustArtifactPath {
        // Used by the Rust release workflow to validate the freshly built XCFramework
        // before publishing it as a remote binary artifact.
        .binaryTarget(name: "TokenizersRust", path: localRustArtifactPath)
    } else {
        .binaryTarget(
            name: "TokenizersRust",
            url: "https://github.com/DePasqualeOrg/swift-tokenizers/releases/download/tokenizers-rust-0.3.1/TokenizersRust-0.3.1.xcframework.zip",
            checksum: "d25288b933b3aa164b661f8ff2b02a53c57c9b32a20dd1ac8a6c3429467f6fcd"
        )
    }

var packageTargets: [Target] = [
    tokenizersRustTarget,
    .target(
        name: "TokenizersCore",
        dependencies: [],
        path: "Sources/Tokenizers",
        exclude: excludedTokenizerSources(keeping: tokenizerCoreSources),
        sources: tokenizerCoreSources
    ),
    .target(
        name: "TokenizersSwiftBackend",
        dependencies: [
            "TokenizersCore",
            .product(name: "Jinja", package: "swift-jinja", condition: .when(traits: ["Swift"])),
            .product(name: "yyjson", package: "yyjson", condition: .when(traits: ["Swift"])),
        ],
        path: "Sources/Tokenizers",
        exclude: excludedTokenizerSources(keeping: tokenizerSwiftBackendSources),
        sources: tokenizerSwiftBackendSources,
        swiftSettings: [
            .define("TOKENIZERS_SWIFT_BACKEND", .when(traits: ["Swift"]))
        ]
    ),
    .target(
        name: "TokenizersRustBackend",
        dependencies: [
            "TokenizersCore",
            .target(name: "TokenizersRust", condition: .when(traits: ["Rust"])),
        ],
        path: "Sources/Tokenizers",
        exclude: excludedTokenizerSources(keeping: tokenizerRustBackendSources),
        sources: tokenizerRustBackendSources,
        swiftSettings: [
            .define("Rust", .when(traits: ["Rust"]))
        ]
    ),
    .target(
        name: "Tokenizers",
        dependencies: [
            "TokenizersCore",
            .target(name: "TokenizersSwiftBackend", condition: .when(traits: ["Swift"])),
            .target(name: "TokenizersRustBackend", condition: .when(traits: ["Rust"])),
        ],
        path: "Sources/TokenizersFacade",
        swiftSettings: [
            .define("TOKENIZERS_SWIFT_BACKEND", .when(traits: ["Swift"])),
            .define("Rust", .when(traits: ["Rust"])),
        ]
    ),
    .testTarget(
        name: "TokenizersTests",
        dependencies: [
            "Tokenizers",
            "TokenizersCore",
            .target(name: "TokenizersSwiftBackend", condition: .when(traits: ["Swift"])),
            .target(name: "TokenizersRustBackend", condition: .when(traits: ["Rust"])),
            .product(name: "HFAPI", package: "swift-hf-api"),
        ],
        resources: [.process("Resources")],
        swiftSettings: [
            .define("TOKENIZERS_SWIFT_BACKEND", .when(traits: ["Swift"])),
            .define("Rust", .when(traits: ["Rust"])),
        ]
    ),
]

if benchmarksEnabled {
    packageTargets.append(
        .testTarget(
            name: "Benchmarks",
            dependencies: [
                "Tokenizers",
                "TokenizersCore",
                .target(name: "TokenizersSwiftBackend", condition: .when(traits: ["Swift"])),
                .target(name: "TokenizersRustBackend", condition: .when(traits: ["Rust"])),
                .product(name: "HFAPI", package: "swift-hf-api"),
                .product(name: "BenchmarkHelpers", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            swiftSettings: [
                .define("TOKENIZERS_SWIFT_BACKEND", .when(traits: ["Swift"])),
                .define("Rust", .when(traits: ["Rust"])),
            ]
        )
    )
}

let package = Package(
    name: "swift-tokenizers",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Tokenizers", targets: ["Tokenizers"])
    ],
    traits: [
        .default(enabledTraits: ["Swift"]),
        .trait(name: "Swift"),
        .trait(name: "Rust"),
    ],
    dependencies: packageDependencies,
    targets: packageTargets
)
