# Lilims justfile
# Common development tasks for the Lilims project
# Uses system Swift/Xcode toolchain via Nix environment

# Clean build artifacts
clean:
    rm -rf .build
    rm -rf DerivedData
    swift package clean
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + || true
    rm -rf .pytest_cache
    rm -rf .ruff_cache

# Build the project using system Swift
build:
    swift build

# Run all tests (Swift via system toolchain + Python)
test:
    swift test
    pytest Scripts/tests

# Run linting (Python via ruff + Swift tests)
lint:
    ruff check Scripts
    swift test
