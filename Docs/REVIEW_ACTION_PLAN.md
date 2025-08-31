# Review Action Plan for WS-1 and WS-2

## Critical Issues Found

### WS-1 CoreMLConversion
1. **GGUF conversion is non-functional** - Only demonstrates with first tensor, doesn't convert actual model
2. **Using deprecated CoreML API** - NeuralNetworkBuilder instead of ML Program format
3. **Incorrect quantization parameters** - Using invalid `precision="int4"` parameter
4. **No model architecture support** - Cannot handle different transformer types
5. **Manifest generator missing** - No `manifest.py` implementation

### WS-2 RuntimeCoreML  
1. **Model interface mismatch** - Backend expects features that conversion doesn't provide
2. **KV cache broken** - Only stores last token, not full sequence
3. **No transformer implementation** - Missing actual attention/transformer logic
4. **Rope utilities unused** - Created but not integrated
5. **No model validation** - Doesn't verify loaded model compatibility

## Priority 1: Fix Model Conversion Pipeline

### Task 1.1: Implement proper CoreML conversion
- [x] Use coremltools ML Program format with stateful models
- [x] Support flexible input shapes for sequences
- [x] Implement proper INT4 weight quantization
- [x] Add model architecture detection (GPT-2, Phi-2, Gemma)

### Task 1.2: Create manifest generator
- [x] Implement `Scripts/manifest.py` to generate model metadata
- [x] Include model name, size, SHA256, runtime version
- [x] Add semantic versioning support

### Task 1.3: Fix GGUF conversion
- [ ] Parse GGUF model architecture properly
- [ ] Map GGUF tensors to CoreML layers
- [ ] Handle different quantization formats (Q4_0, Q4_K, etc.)

## Priority 2: Fix Runtime Backend

### Task 2.1: Implement stateful transformer model
- [ ] Create proper transformer architecture in CoreML
- [ ] Add multi-head attention with KV caching
- [ ] Integrate rotary position embeddings
- [ ] Support autoregressive generation

### Task 2.2: Fix KV cache implementation
- [ ] Store full sequence history, not just last token
- [ ] Implement proper cache paging/eviction
- [ ] Support variable sequence lengths
- [ ] Add memory management

### Task 2.3: Align model interface
- [ ] Ensure conversion produces models with expected inputs/outputs
- [ ] Add model validation in backend initialization
- [ ] Handle different model architectures uniformly

## Priority 3: Testing & Validation

### Task 3.1: Create test models
- [ ] Convert TinyStories model for testing
- [ ] Add test fixtures with known outputs
- [ ] Create minimal test cases for each component

### Task 3.2: Fix perplexity evaluation
- [ ] Align evaluation with actual model interface
- [ ] Add proper tokenization handling
- [ ] Compare INT4 vs FP16 properly

### Task 3.3: Add integration tests
- [ ] Test full pipeline: conversion → loading → generation
- [ ] Verify memory usage stays within bounds
- [ ] Benchmark tokens/second performance

## Priority 4: Documentation & Code Quality

### Task 4.1: Update documentation
- [ ] Document actual conversion workflow that works
- [ ] Add examples for each supported model type
- [ ] Include troubleshooting guide

### Task 4.2: Improve error handling
- [ ] Add meaningful error messages for common failures
- [ ] Validate inputs before processing
- [ ] Handle edge cases gracefully

## Implementation Order

1. **Week 1**: Fix model conversion (Tasks 1.1, 1.2)
2. **Week 2**: Implement stateful transformer (Task 2.1)  
3. **Week 3**: Fix KV cache and interface alignment (Tasks 2.2, 2.3)
4. **Week 4**: Testing and validation (Tasks 3.1-3.3)
5. **Week 5**: Documentation and polish (Tasks 4.1-4.2)

## Success Criteria

- [ ] Can convert at least one real model (e.g., GPT-2 125M)
- [ ] Model loads and generates coherent text
- [ ] Perplexity delta < 3% vs FP16 baseline
- [ ] All tests pass in CI
- [ ] Memory usage < 1GB for 8k context
- [ ] Tokens/second > 10 on A17 Pro

## Next Steps

1. Start with fixing the conversion pipeline to use proper ML Program format
2. Create a minimal working transformer model
3. Iterate on performance and accuracy
4. Add support for more model architectures