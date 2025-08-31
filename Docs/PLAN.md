## Executive summary  

Ship a **chat‑style iOS app** whose core is a *Swift* inference runtime able to load any small transformer that has been (a) quantized to 4–8 bit and (b) converted to **Core ML**. Core ML heuristics dispatch ops across ANE, GPU, and CPU—we profile but cannot force ANE placement. The plan focuses on a single Core ML backend with an 8 k context window.  Apple’s recent WWDC 24 sessions, Core ML documentation, and community projects (swift‑coreml‑transformers, ggml/gguf) provide all the building blocks and performance data you need.  [oai_citation:0‡Apple Developer](https://developer.apple.com/videos/play/wwdc2024/10161/?utm_source=chatgpt.com) [oai_citation:1‡Apple Developer](https://developer.apple.com/machine-learning/core-ml/?utm_source=chatgpt.com) [oai_citation:2‡Apple Developer](https://developer.apple.com/documentation/metalperformanceshadersgraph?utm_source=chatgpt.com) [oai_citation:3‡Apple Developer](https://developer.apple.com/videos/play/wwdc2024/10211/?utm_source=chatgpt.com) [oai_citation:4‡Apple Developer](https://developer.apple.com/documentation/accelerate/bnns?utm_source=chatgpt.com) [oai_citation:6‡GitHub](https://github.com/klyap/swift-coreml-transformers-demo?utm_source=chatgpt.com) [oai_citation:7‡GitHub](https://github.com/ggml-org/llama.cpp/discussions/4423?utm_source=chatgpt.com) [oai_citation:8‡GitHub](https://github.com/hollance/neural-engine/blob/master/docs/ane-vs-gpu.md?utm_source=chatgpt.com) [oai_citation:9‡arXiv](https://arxiv.org/html/2403.12844?utm_source=chatgpt.com)

---

## 1. Goals & constraints  

| Requirement | Notes |
|-------------|-------|
| **Offline, private** | All inference on‑device; no network calls for generation. |
| **Sub‑3 s first‑token latency on A17 Pro** | Use ANE‑first scheduling and 4‑bit weight quantization.  [oai_citation:10‡apple.github.io](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html?utm_source=chatgpt.com) [oai_citation:11‡Apple Machine Learning Research](https://machinelearning.apple.com/research/core-ml-on-device-llama?utm_source=chatgpt.com) |
| **<1 GB total disk per model** | Enforce gguf/ggml or Core ML compressed weights. |
| **Extensible** | Must load any future 2‑8 B model with minimal work. |
| **AI‑agent friendly repo** | Each workstream has its own sub‑folder, issue template, and test target. |

---

## 2. High‑level architecture  

```
App
 ├─ UI (SwiftUI)
 ├─ ModelManager              (downloads, versioning, disk quotas)
 ├─ Runtime                   (CoreMLBackend)
 │    └─ Tokenizer            (Swift port of HF tokenizer)
 └─ Utilities                 (logging, metrics, telemetry off switch)
```
*The Runtime exposes `generate(prompt:stream:)`, interrupt, and profile APIs.*
[oai_citation:12‡GitHub](https://github.com/StanfordSpezi/SpeziLLM?utm_source=chatgpt.com)

---

## 3. Model strategy  

1. **Pick baselines**  
   * Phi‑2 (2.7 B)  
   * Gemma‑2B‑It  
   * TinyStories‑1M for unit tests.  [oai_citation:13‡GitHub](https://github.com/klyap/swift-coreml-transformers-demo?utm_source=chatgpt.com)  

2. **Convert to Core ML**  
   ```python
   import coremltools as ct
   mlmodel = ct.convert(torch_model,
                        compute_units="ALL",
                        compute_precision=ct.precision.INT4_WEIGHT)
   mlmodel.save("phi2-int4.mlpackage")
   ```  
   Follow Apple’s PyTorch conversion workflow; enable *weight‑only* INT4 plus activation FP16.  [oai_citation:14‡apple.github.io](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html?utm_source=chatgpt.com) [oai_citation:15‡apple.github.io](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html?utm_source=chatgpt.com) [oai_citation:16‡apple.github.io](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html?utm_source=chatgpt.com)

3. **Package format**
   Store models as `.mlpackage` plus a side‑car JSON manifest (name, size, SHA256, **semver**, compatible runtime version).  Provide a migration script for gguf→mlpackage so users can import community checkpoints.  [oai_citation:17‡GitHub](https://github.com/ggml-org/llama.cpp/discussions/4423?utm_source=chatgpt.com)

## 3.5 Quantization validation

1. **Baseline measurements required**
   - FP16 baseline perplexity on your eval set
   - INT8 degradation (should be <1%)
   - INT4 degradation (expect 3-5%)
   - Actual tokens/sec on target hardware with each precision

2. **Core ML quantization limitations**
   - Weight-only quantization (activations stay FP16)
   - Limited op support in INT4 mode
   - Some layers may silently fall back to FP32

---

## 4. Inference engine design (CoreMLBackend)

| Layer | Key decisions | Reality check |
|-------|---------------|--------------|
| **Graph execution** | Use Core ML async prediction API | ANE scheduling is opaque - profile but don't expect control |
| **Custom ops** | Avoid custom ops initially - use only Core ML native ops | MPSGraph custom ops will likely run on GPU, defeating ANE purpose |
| **Memory management** | Start with 8k context, not 32k | Calculate actual memory: (2 * num_layers * seq_len * hidden_dim * 2 bytes) |
| **Fallback strategy** | Single backend only - Core ML or llama.cpp, not both | Pick one and optimize it properly |

---

## 5. Backend strategy

Choose ONE:
- **Option A**: Core ML only - maximize Apple platform integration
- **Option B**: llama.cpp only - maximize model compatibility

Don't build both. The switching abstraction adds complexity for no user benefit.

---

## 6. Swift implementation workstreams

Cross-cutting concerns:
- **Error handling**: surface model load failures and OOM mid-generation gracefully.
- **Background execution**: request `beginBackgroundTask` during long generations.
- **Model versioning**: manifest semantic versioning; runtime rejects incompatible models.
- **Streaming backpressure**: use bounded `AsyncSequence` to avoid dropped tokens.
- **Testing**: unit tests for tokenizer, context window, and error paths; integration tests for generation and memory limits.

| ID | Folder | Tasks (can run in parallel) |
|----|--------|-----------------------------|
| **WS‑1 CoreMLConversion** | Scripts & notebooks for model -> `.mlpackage`; unit tests validate perplexity drop ≤ 3 % vs FP16. |
| **WS‑2 RuntimeCoreML** | Build token loop, streaming callback, KV cache tensors, rope & rotary tables. |
| **WS‑3 ContextWindow** | Implement sliding window attention, memory-mapped weight loading, batch prefill optimization |
| **WS‑4 Tokenizer** | Port GPT‑2 BPE to Swift + SIMD, fuzz against HuggingFace tokenizer.json. |
| **WS‑5 ModelManager** | Async download with checksum, iCloud exclusion, LRU purge. |
| **WS‑6 BenchmarkKit** | XCUITests measuring tokens/s and energy via `metricKit`. |
| **WS‑7 UI** | SwiftUI chat, model picker, live tokens stream, temperature slider. |
| **WS‑8 Telemetry** | Optional opt‑in analytics (device, tokens/s, crashes) stored locally. |
| **WS‑9 Docs** | Auto‑publish API doc with DocC; diagrams via Graphviz. |
| **WS‑10 CI/CD** | GitHub Actions: build, unit + integration tests on iPhone‑15, artifact upload to TestFlight. |
| **WS‑11 AgentScripts** | YAML specs so code‑gen agents can claim tasks and commit PRs. |
### WS-1 CoreMLConversion tasks
- [x] **Create `convert.py`** – convert gguf or PyTorch checkpoints to `.mlpackage` with INT4 weights.
- [x] **Write `manifest.json` generator** – store name, size, SHA256 and runtime version.
- [x] **Provide `evaluate_perplexity.py`** – compare quantized perplexity to FP16 on TinyStories.
- [x] **Add CI unit test** – fail if perplexity increases more than 3%.
- [x] **Document conversion workflow** in `Docs/ConversionGuide.md`.
- [ ] **Finalize `--gguf` conversion** – current script only validates the GGUF header; implement real weight loading and Core ML export.
- [ ] **Add TinyStories dataset helper** – download and cache the evaluation corpus for `evaluate_perplexity.py`.
- [ ] **Improve error handling** – detect missing dependencies and report user‑friendly CLI errors.


### WS-2 RuntimeCoreML tasks
- [x] **Create `CoreMLBackend.swift`** – token loop using stateful model evaluation.
- [x] **Expose `TokenStreamDelegate`** – callback for each generated token.
- [x] **Implement KV cache tensors** with `MLShapedArray` and LRU paging.
- [x] **Provide rope & rotary table kernels** via `MPSGraph` fallback.
- [x] **Support backpressure-aware streaming** with `AsyncSequence`
- [x] **Handle OOM errors** and surface to callers
- [ ] **Sample from logits** – convert model logits to tokens with adjustable temperature/top‑k instead of relying on a precomputed `token` feature.
- [ ] **Feed KV cache back into model inputs** so past context is reused across calls.
- [ ] **Add cancellation support** for long‑running generation.
- [ ] **Unit test** decoding 20 tokens from a TinyStories checkpoint.

### WS-3 ContextWindow tasks
- [ ] **Implement sliding window attention**
- [ ] **Add memory-mapped weight loading**
- [ ] **Optimize batch prefill**



---

## 7. Performance & profiling  

* Use **Xcode Instruments → Core ML report** to verify operator placement.  [oai_citation:26‡Apple Developer](https://developer.apple.com/machine-learning/core-ml/?utm_source=chatgpt.com)  
* Compare ANE vs GPU paths with the *neural‑engine* community benchmarks.  [oai_citation:27‡GitHub](https://github.com/hollance/neural-engine/blob/master/docs/ane-vs-gpu.md?utm_source=chatgpt.com)  
* Track tokens/s, energy (mWh), and peak RSS; fail CI if regression >10 %.  

---

## 8. UX & product guardrails  

* **Startup wizard**: offer default 400 MB TinyStories and let users download larger checkpoints.  
* **Safety**: embed an optional local moderation classifier; disable on‑device if user is under 16 (age‑gate).  
* **Accessibility**: live VoiceOver announcements for streamed tokens.
* **Background execution**: request background task time for long generations.

---

## 9. Privacy & security checklist  

1. Run in **app sandbox**; no disk read outside `Application Support/Lilims`.  
2. Entitlements: only `com.apple.security.files.user-selected.read-write` for model import.  
3. Do *not* request network permission by default; only when user adds a remote model URL.  
4. Hardened runtime; enable code integrity auditing in build settings.  

---

## 10. DevOps & automation  

* **Repo layout**  
  ```
  /Scripts         (python conversion & quantization)
  /Sources
     /Runtime
        /CoreML
        /Tokenizer
     /UI
  /Tests           (unit)
  /BenchmarkKit    (integration perf tests)
  /Docs
  *.xcworkspace
  ```
* **Issue templates** auto‑label workstream IDs.  
* **Pull‑request bot**: run `swift test`, `swiftlint`, and Benchmarks; refuse merges if tokens/s drops.  

---

## 11. Milestones (suggested 20‑week schedule)

| Weeks | Deliverable |
|------|-------------|
| 1‑2 | Phi‑2 converted, TinyStories unit tests passing (WS‑1,4) |
| 3‑4 | CoreMLBackend generates single token, benchmark harness (WS‑2,6) |
| 5‑6 | Streamed generation demo in console |
| 7‑8 | SwiftUI chat UI & model picker (WS‑7) |
| 9‑10 | ContextWindow features: sliding attention, memory-mapped weights (WS‑3) |
| 11‑12 | ModelManager downloads & storage quota (WS‑5) |
| 13‑14 | Energy profiling, KV‑cache paging (WS‑2) |
| 15‑16 | Accessibility & moderation (WS‑7) |
| 17‑18 | Telemetry opt‑in, crash reporting, CI gates (WS‑8,10) |
| 19‑20 | Beta build on TestFlight, dog‑food and iterate |

---

## 12. Pre-launch validation gates

Before any TestFlight release:
1. Memory profiling on 6GB RAM device (iPhone 12) with 8k context
2. Thermal throttling test: 5-minute continuous generation
3. Actual tokens/sec measurement vs. claims
4. Binary size with embedded model (must be <200MB for OTA updates)
