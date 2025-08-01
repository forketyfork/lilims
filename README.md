# Lilims

Lilims is an iOS application project focused on running small language models entirely on device. The app can download, convert and execute LLMs using Core ML or llama.cpp backends. The repository now contains a minimal Swift Package that builds an iOS app using Swift 6.0.

```
Scripts/        Python helpers for model conversion and quantization (Python 3.13)
Sources/        Swift code for the app and runtime libraries
    Runtime/
        CoreML/   Core ML pipeline and model management
        LlamaCpp/ Interface to the llama.cpp backend
        Tokenizer/Swift tokenizer utilities
    UI/          SwiftUI-based user interface
Tests/          Unit tests for app functionality
BenchmarkKit/   Integration performance tests
Docs/           Additional documentation
*.xcworkspace   Xcode workspace referencing the Swift package
```

The goal is to experiment with on-device LLM inference and benchmarking on Apple silicon. The runtime uses Swift 6.0 and the scripts require Python 3.13. Continuous integration builds the iOS target and runs the Swift and Python tests on each pull request. The iOS workflow installs Swift 6.1 using [`swift-actions/setup-swift`](https://github.com/swift-actions/setup-swift) and invokes `xcodebuild` with `-toolchain "$TOOLCHAINS"` so the correct toolchain is used.
Python scripts are linted with [Ruff](https://docs.astral.sh/ruff/) and tested using `pytest`.


## Python environment

The Python helpers use a virtual environment. Create one and install the
required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

CI uses the same steps so local runs match the GitHub build.
