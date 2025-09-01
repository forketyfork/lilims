# Lilims justfile
# Common development tasks for the Lilims project

# Clean build artifacts
clean:
    rm -rf .build
    rm -rf DerivedData
    swift package clean
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + || true

# Build the project
build:
    swift build

# Run all tests (Swift and Python)
test:
    swift test
    pytest Scripts/tests

# Run linting (Swift and Python)
lint:
    ruff check Scripts
    swift test