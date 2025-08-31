## Glossary

### Apple‑specific runtime & tooling

| Term | Meaning | Why it matters |
|------|---------|---------------|
| **Apple Neural Engine (ANE)** | Dedicated ML accelerator in Apple Silicon, optimized for convolutions and some transformer ops. Core ML *may* use it based on operator compatibility and heuristics. | Can provide 1.5-3× speedup over GPU for compatible ops, but you cannot force or guarantee ANE usage. |
| **Core ML Compute Units (`MLComputeUnits`)** | An enum that tells Core ML which hardware blocks (CPU, GPU, ANE, or *all*) it may use at runtime. | Your code sets `.all` in production, but CI can force `.cpuOnly` to catch hidden device‑specific bugs. |
| **Unified Memory** | Apple Silicon's shared RAM between CPU/GPU/ANE. No PCIe copies needed. | Why iOS can run larger models than equivalent PC RAM would suggest, but still has hard limits. |

### Model formats & tooling

| Term | Meaning | Why it matters |
|------|---------|---------------|
| **ggml** | A tensor library and file format created for *llama.cpp*; stores weights in a single binary with optional quantization.  [oai_citation:11‡GitHub](https://github.com/ggml-org/llama.cpp?utm_source=chatgpt.com) | Baseline for CPU‑centric inference; easy to port to Swift via C FFI. |
| **gguf** | *GGML Universal Format*—successor to ggml that adds richer metadata and broader model support beyond Llama.  [oai_citation:12‡MLK - Machine Learning Knowledge](https://machinelearningknowledge.ai/gguf-vs-ggml-understanding-the-differences/?utm_source=chatgpt.com) [oai_citation:13‡DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/6.1-gguf-file-format?utm_source=chatgpt.com) | Preferred intermediate when importing community checkpoints; conversion scripts handle gguf → Core ML. |
| **llama.cpp** | C/C++ reference implementation for quantized Llama‑family models; runs on CPU, GPU, or Metal.  [oai_citation:14‡GitHub](https://github.com/ggml-org/llama.cpp?utm_source=chatgpt.com) | Provides an alternate backend for benchmarking against your Core ML path. |
| **Core ML .mlpackage** | Apple's compiled model format. Contains weight data, compute graph, and metadata. | Not just weights - includes full graph definition. Can be 2-3× larger than equivalent GGUF. |

### Quantization & numeric formats

| Term | Meaning | Why it matters |
|------|---------|---------------|
| **FP16 (half‑precision)** | IEEE‑754 16‑bit floating‑point format; 1 sign, 5 exponent, 10 mantissa bits.  [oai_citation:15‡Wikipedia](https://en.wikipedia.org/wiki/Half-precision_floating-point_format?utm_source=chatgpt.com) | Standard activation precision on ANE/GPU—halves memory bandwidth relative to FP32 with minimal accuracy loss. |
| **INT4 weight‑only quantization (WOQ)** | Compresses just the *weights* to 4‑bit integers while keeping activations in FP16/FP32; minimizes memory‑bandwidth bottlenecks during GEMMs.  [oai_citation:16‡apple.github.io](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html?utm_source=chatgpt.com) [oai_citation:17‡NVIDIA Docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html?utm_source=chatgpt.com) | Critical to hit sub‑3 s first‑token latency on‑device without killing accuracy. |
| **Activation quantization** | Quantizing intermediate tensors, not just weights. Core ML doesn't support this for transformers. | Why your "INT4" model still uses 16-bit memory for activations and KV cache. |

### Transformer inference internals

| Term | Meaning | Why it matters |
|------|---------|---------------|
| **KV cache** | Stores attention keys and values for past tokens. Memory usage = 2 × layers × seq_len × hidden_dim × activation_precision (typically FP16). | Primary memory consumer during inference. 8k context for 2.7B model ≈ 2.7 GB in FP16. |
| **KV cache allotment** | Pre-allocated memory blocks for KV storage. Must account for activation precision (FP16), not weight quantization. | Example: 8k context window requires ~2.7 GB for Phi-2, regardless of INT4 weight compression. |
| **RoPE / rotary tables** | *Rotary Position Embedding*: multiplies query and key vectors by sinusoidal rotations that encode absolute positions.  [oai_citation:19‡arXiv](https://arxiv.org/abs/2104.09864?utm_source=chatgpt.com) | Supports context extension and scales better than vanilla sine/cosine; requires a custom Metal/MPS kernel on ANE. |
| **Perplexity** | Cross‑entropy exponentiation metric that measures how “surprised” a language model is; lower = better fit.  [oai_citation:20‡GeeksforGeeks](https://www.geeksforgeeks.org/nlp/perplexity-for-llm-evaluation/?utm_source=chatgpt.com) | Unit tests in WS‑1 must limit perplexity drop to ≤ 3 % after quantization. |
| **Tokens per second** | Throughput metric. Modern Apple Silicon: M3 Max ~80 t/s (7B INT4), A17 Pro ~15‑20 t/s (2B INT4), older A15 ~5‑8 t/s. | Set realistic targets based on actual hardware, not theoretical calculations. |

### Miscellaneous

| Term | Meaning | Why it matters |
|------|---------|---------------|
| **Streaming generation** | Emitting tokens incrementally via a callback rather than waiting for an entire sequence. | Reduces perceived latency; enables UI token‑by‑token display. |
| **BPE tokenizer** | *Byte‑Pair Encoding*: iterative merge‑rules that map UTF‑8 bytes to sub‑word tokens; adopted by GPT‑2 and most modern LLMs. | Your Swift SIMD tokenizer must round‑trip byte‑level merges exactly to match server checkpoints. |
| **Thermal throttling** | Performance degradation when chip overheats. Kicks in after ~30 seconds of sustained inference on iPhone. | Why your "50 t/s" becomes 10 t/s after a minute. Must test sustained performance. |

---

### Quick reality checks

* **INT4 quantization is weights-only** - Your KV cache and activations remain FP16
* **ANE is not programmable** - You get what Core ML gives you
* **Memory calculation** - Weights + KV cache + iOS overhead = add 50% buffer
* **Sustained performance** - Whatever tokens/s you measure, divide by 2 for thermal reality
