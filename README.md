# Lilims

Lilims is an iOS application project focused on running small language models entirely on device. The app can download, convert and execute LLMs using Core ML or llama.cpp backends. This repository currently contains only the project scaffold.

```
Scripts/        Python helpers for model conversion and quantization
Sources/        Swift code for the app and runtime libraries
    Runtime/
        CoreML/   Core ML pipeline and model management
        LlamaCpp/ Interface to the llama.cpp backend
        Tokenizer/Swift tokenizer utilities
    UI/          SwiftUI-based user interface
Tests/          Unit tests for app functionality
BenchmarkKit/   Integration performance tests
Docs/           Additional documentation
*.xcworkspace   Xcode workspace to open in Xcode
```

The goal is to experiment with on-device LLM inference and benchmarking on Apple silicon. The runtime will expose both Core ML and llama.cpp pathways with a tokenizer written in Swift.

