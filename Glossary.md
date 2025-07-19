## Glossary

### Apple‑specific runtime & tooling

| Term | Meaning | Why it matters |
|------|---------|---------------|
| **Apple Neural Engine (ANE)** | A block of dedicated matrix‑multiply cores inside A‑series and M‑series SoCs, optimized for low‑power ML inference. Models scheduled to ANE via Core ML often unlock ×10 throughput and lower peak memory vs. GPU or CPU.  [oai_citation:0‡Apple Machine Learning Research](https://machinelearning.apple.com/research/neural-engine-transformers?utm_source=chatgpt.com) [oai_citation:1‡Apple Wiki](https://apple.fandom.com/wiki/Neural_Engine?utm_source=chatgpt.com) | Your pipeline’s default “fast path”; measure operator placement to be sure you’re actually hitting it. |
| **BNNS / BNNSGraph** | *Basic Neural‑Network Subroutines*—part of the Accelerate framework; BNNS offers single‑op kernels (conv, matmul) whereas BNNSGraph stitches them into a compute graph you can execute on CPU.  [oai_citation:2‡Apple Developer](https://developer.apple.com/documentation/accelerate/bnns?utm_source=chatgpt.com) [oai_citation:3‡Apple Developer](https://developer.apple.com/documentation/accelerate/bnnsgraph?utm_source=chatgpt.com) | Provides a last‑resort, deterministic CPU fallback when an op isn’t supported on ANE or GPU. |
| **Core ML Compute Units (`MLComputeUnits`)** | An enum that tells Core ML which hardware blocks (CPU, GPU, ANE, or *all*) it may use at runtime.  [oai_citation:4‡Apple Developer](https://developer.apple.com/documentation/coreml/mlcomputeunits?utm_source=chatgpt.com) | Your code sets `.all` in production, but CI can force `.cpuOnly` to catch hidden device‑specific bugs. |
| **MPSGraph** | The Metal Performance Shaders *Graph* API: a symbolic compute graph that runs on the GPU (and can spill to ANE) with compiler optimizations like fusion and dead‑code elimination.  [oai_citation:5‡Apple Developer](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph?utm_source=chatgpt.com) [oai_citation:6‡Apple Developer](https://developer.apple.com/documentation/metalperformanceshadersgraph?utm_source=chatgpt.com) | Lets you implement missing transformer ops (e.g., RoPE slice) without diving into Metal shaders. |
| **MetricKit** | iOS framework that delivers per‑device, privacy‑safe reports on energy, CPU, and crash metrics.  [oai_citation:7‡Apple Developer](https://developer.apple.com/documentation/metrickit?utm_source=chatgpt.com) [oai_citation:8‡uptech.team](https://www.uptech.team/blog/how-to-measure-app-performance-with-metrickit?utm_source=chatgpt.com) | Use it to regress “tokens /s” and mWh in the BenchmarkKit test target; fail CI on a 10 % energy regression. |
| **DocC** | The Swift documentation compiler; it turns Markdown comments plus tutorials into browsable docs in Xcode or a static site.  [oai_citation:9‡Swift.org](https://www.swift.org/documentation/docc/?utm_source=chatgpt.com) [oai_citation:10‡Apple Developer](https://developer.apple.com/documentation/xcode/writing-documentation?utm_source=chatgpt.com) | Every runtime protocol and public struct in the repo should ship DocC comments so agents get instant API docs in Xcode Quick Help. |

### Model formats & tooling

| Term | Meaning | Why it matters |
|------|---------|---------------|
| **ggml** | A tensor library and file format created for *llama.cpp*; stores weights in a single binary with optional quantization.  [oai_citation:11‡GitHub](https://github.com/ggml-org/llama.cpp?utm_source=chatgpt.com) | Baseline for CPU‑centric inference; easy to port to Swift via C FFI. |
| **gguf** | *GGML Universal Format*—successor to ggml that adds richer metadata and broader model support beyond Llama.  [oai_citation:12‡MLK - Machine Learning Knowledge](https://machinelearningknowledge.ai/gguf-vs-ggml-understanding-the-differences/?utm_source=chatgpt.com) [oai_citation:13‡DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/6.1-gguf-file-format?utm_source=chatgpt.com) | Preferred intermediate when importing community checkpoints; conversion scripts handle gguf → Core ML. |
| **llama.cpp** | C/C++ reference implementation for quantized Llama‑family models; runs on CPU, GPU, or Metal.  [oai_citation:14‡GitHub](https://github.com/ggml-org/llama.cpp?utm_source=chatgpt.com) | Provides an alternate backend for benchmarking against your Core ML path. |

### Quantization & numeric formats

| Term | Meaning | Why it matters |
|------|---------|---------------|
| **FP16 (half‑precision)** | IEEE‑754 16‑bit floating‑point format; 1 sign, 5 exponent, 10 mantissa bits.  [oai_citation:15‡Wikipedia](https://en.wikipedia.org/wiki/Half-precision_floating-point_format?utm_source=chatgpt.com) | Standard activation precision on ANE/GPU—halves memory bandwidth relative to FP32 with minimal accuracy loss. |
| **INT4 weight‑only quantization (WOQ)** | Compresses just the *weights* to 4‑bit integers while keeping activations in FP16/FP32; minimizes memory‑bandwidth bottlenecks during GEMMs.  [oai_citation:16‡apple.github.io](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html?utm_source=chatgpt.com) [oai_citation:17‡NVIDIA Docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html?utm_source=chatgpt.com) | Critical to hit sub‑3 s first‑token latency on‑device without killing accuracy. |

### Transformer inference internals

| Term | Meaning | Why it matters |
|------|---------|---------------|
| **KV cache (Key/Value cache)** | Stores attention keys *K* and values *V* for all past tokens so the model recomputes only the new query *Q* each step.  [oai_citation:18‡Hugging Face](https://huggingface.co/blog/not-lain/kv-caching?utm_source=chatgpt.com) | Slashes decode FLOPs from O(T²) to O(T) and is the main memory consumer; see “KV cache allotment” below. |
| **KV cache allotment / paging** | Strategy for reserving and evicting chunks of KV memory (e.g., LRU) to bound RAM usage—32 k tokens of INT4 weights ≈ 512 MB. | Prevents out‑of‑memory crashes on older devices while still allowing long conversations. |
| **RoPE / rotary tables** | *Rotary Position Embedding*: multiplies query and key vectors by sinusoidal rotations that encode absolute positions.  [oai_citation:19‡arXiv](https://arxiv.org/abs/2104.09864?utm_source=chatgpt.com) | Supports context extension and scales better than vanilla sine/cosine; requires a custom Metal/MPS kernel on ANE. |
| **Perplexity** | Cross‑entropy exponentiation metric that measures how “surprised” a language model is; lower = better fit.  [oai_citation:20‡GeeksforGeeks](https://www.geeksforgeeks.org/nlp/perplexity-for-llm-evaluation/?utm_source=chatgpt.com) | Unit tests in WS‑1 must limit perplexity drop to ≤ 3 % after quantization. |
| **Tokens per second (t/s)** | Throughput metric: number of generated tokens emitted per wall‑clock second. Benchmarks often show 50–150 t/s on a desktop GPU; Apple silicon M‑series can reach 25 t/s for 7‑B models.  [oai_citation:21‡GitHub](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference?utm_source=chatgpt.com) [oai_citation:22‡NVIDIA Developer](https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/?utm_source=chatgpt.com) | Primary KPI for user experience and CI performance gates. |

### Miscellaneous

| Term | Meaning | Why it matters |
|------|---------|---------------|
| **Streaming generation** | Emitting tokens incrementally via a callback rather than waiting for an entire sequence. | Reduces perceived latency; enables UI token‑by‑token display. |
| **BPE tokenizer** | *Byte‑Pair Encoding*: iterative merge‑rules that map UTF‑8 bytes to sub‑word tokens; adopted by GPT‑2 and most modern LLMs. | Your Swift SIMD tokenizer must round‑trip byte‑level merges exactly to match server checkpoints. |

---

### Quick cheat‑sheet for newcomers

* **If you see “graph” think hardware routing** (Core ML → ANE/GPU, MPSGraph → GPU, BNNSGraph → CPU).  
* **If you see “4‑bit” think WOQ + ANE**: Core ML can load 4‑bit weights directly; the GPU path can’t yet.  
* **If you see “KV” think RAM pressure**—monitor virtual‑memory spikes in Instruments.