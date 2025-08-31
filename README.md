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


## Development Environment

### Option 1: Nix + direnv (Recommended)

The project includes a Nix flake that provides a complete, reproducible development environment with all dependencies:

```bash
# Install Nix (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install

# Install direnv (if not already installed)
nix profile install nixpkgs#direnv

# Enable direnv in your shell (add to ~/.bashrc or ~/.zshrc)
eval "$(direnv hook bash)"  # for bash
eval "$(direnv hook zsh)"   # for zsh

# Clone and enter the project directory
cd lilims
direnv allow
```

The environment includes:
- Swift 6.0+ toolchain with swiftformat and swiftlint
- Python 3.13 with numpy, pytest, and ruff
- Development utilities: ripgrep (rg), fd, tree, gh
- Helpful aliases: `test-swift`, `test-python`, `lint-python`, `lint-all`

### Option 2: Python Virtual Environment

Alternatively, you can set up just the Python environment manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

**Note:** You'll still need Swift 6.0+ installed separately for this option.
