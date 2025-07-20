## Executive summary  

Ship a **chat‑style iOS app** whose core is a *Swift* inference runtime able to load any small transformer that has been (a) quantized to 4–8 bit and (b) converted to **Core ML**. The runtime first tries to schedule attention and MLP blocks on ANE through Core ML; if an op is unsupported it falls back to GPU with **MPSGraph**, and finally to CPU with **BNNS**. A thin abstraction layer also lets you swap in the proven **llama.cpp** kernel via a Swift package for experimentation. The plan below stages work in eleven workstreams with explicit deliverables, automated tests, and CI gates.  Apple’s recent WWDC 24 sessions, Core ML documentation, and community projects (SpeziLLM, swift‑coreml‑transformers, ggml/gguf) provide all the building blocks and performance data you need.  [oai_citation:0‡Apple Developer](https://developer.apple.com/videos/play/wwdc2024/10161/?utm_source=chatgpt.com) [oai_citation:1‡Apple Developer](https://developer.apple.com/machine-learning/core-ml/?utm_source=chatgpt.com) [oai_citation:2‡Apple Developer](https://developer.apple.com/documentation/metalperformanceshadersgraph?utm_source=chatgpt.com) [oai_citation:3‡Apple Developer](https://developer.apple.com/videos/play/wwdc2024/10211/?utm_source=chatgpt.com) [oai_citation:4‡Apple Developer](https://developer.apple.com/documentation/accelerate/bnns?utm_source=chatgpt.com) [oai_citation:5‡GitHub](https://github.com/StanfordSpezi/SpeziLLM?utm_source=chatgpt.com) [oai_citation:6‡GitHub](https://github.com/klyap/swift-coreml-transformers-demo?utm_source=chatgpt.com) [oai_citation:7‡GitHub](https://github.com/ggml-org/llama.cpp/discussions/4423?utm_source=chatgpt.com) [oai_citation:8‡GitHub](https://github.com/hollance/neural-engine/blob/master/docs/ane-vs-gpu.md?utm_source=chatgpt.com) [oai_citation:9‡arXiv](https://arxiv.org/html/2403.12844?utm_source=chatgpt.com)

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
 ├─ Runtime                   (protocol)
 │    ├─ CoreMLBackend        (ANE/GPU first)
 │    ├─ LlamaCppBackend      (fallback / benchmarking)
 │    └─ Tokenizer            (Swift port of HF tokenizer)
 └─ Utilities                 (logging, metrics, telemetry off switch)
```

*The Runtime protocol exposes `generate(prompt:stream:)`, interrupt, and profile APIs so UI and tests stay backend‑agnostic.*   [oai_citation:12‡GitHub](https://github.com/StanfordSpezi/SpeziLLM?utm_source=chatgpt.com)

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
   Store models as `.mlpackage` plus a side‑car JSON manifest (name, size, SHA256, compatible runtime version).  Provide a migration script for gguf→mlpackage so users can import community checkpoints.  [oai_citation:17‡GitHub](https://github.com/ggml-org/llama.cpp/discussions/4423?utm_source=chatgpt.com)

---

## 4. Inference engine design (CoreMLBackend)  

| Layer | Key decisions | References |
|-------|---------------|------------|
| **Graph execution** | Use *Core ML Stateful evaluation* so KV‑caches persist on ANE DRAM |  [oai_citation:18‡Apple Developer](https://developer.apple.com/videos/play/wwdc2024/10161/?utm_source=chatgpt.com) |
| **Custom ops** | Implement rope + slice kernels in **MPSGraph** when Core ML lacks them |  [oai_citation:19‡Apple Developer](https://developer.apple.com/documentation/metalperformanceshadersgraph?utm_source=chatgpt.com) [oai_citation:20‡GitHub](https://github.com/ggml-org/llama.cpp/discussions/6871?utm_source=chatgpt.com) |
| **CPU fallback** | BNNSGraph path guarantees determinism for <A14 devices |  [oai_citation:21‡Apple Developer](https://developer.apple.com/documentation/Accelerate/supporting-real-time-ml-inference-on-the-cpu?utm_source=chatgpt.com) [oai_citation:22‡Apple Developer](https://developer.apple.com/documentation/accelerate/bnns?utm_source=chatgpt.com) |
| **Memory management** | Paged KV cache allotment (32 k tokens ≈ 512 MB INT4) with LRU eviction |

Runtime chooses the fastest processor via `MLComputeUnit.all`, then inspects the generated *Core ML performance report* to validate that attention matmuls landed on ANE; if not, it logs an optimization ticket.  [oai_citation:23‡GitHub](https://github.com/hollance/neural-engine/blob/master/docs/ane-vs-gpu.md?utm_source=chatgpt.com)  

---

## 5. Alternative backend: llama.cpp (LlamaCppBackend)  

Embed **ggml** via Swift Package **SpeziLLM**.  This lets power users swap between Metal or CPU kernels and compare throughput without touching UI code. Expose the same Runtime protocol.  [oai_citation:24‡The Swift Package Index](https://swiftpackageindex.com/StanfordSpezi/SpeziLLM?utm_source=chatgpt.com) [oai_citation:25‡GitHub](https://github.com/ggml-org/llama.cpp/discussions/4423?utm_source=chatgpt.com)  

---

## 6. Swift implementation workstreams  

| ID | Folder | Tasks (can run in parallel) |
|----|--------|-----------------------------|
| **WS‑1 CoreMLConversion** | Scripts & notebooks for model -> `.mlpackage`; unit tests validate perplexity drop ≤ 3 % vs FP16. |
| **WS‑2 RuntimeCoreML** | Build token loop, streaming callback, KV cache tensors, rope & rotary tables. |
| **WS‑3 RuntimeLlamaCpp** | Wrap C API, zero‑copy buffers, chunked decode. |
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
- [x] **Implement `--gguf` conversion** – support converting gguf checkpoints to `.mlpackage`.


-### WS-2 RuntimeCoreML tasks
- [x] **Create `CoreMLBackend.swift`** – token loop using stateful model evaluation.
- [ ] **Expose `TokenStreamDelegate`** – callback for each generated token.
- [ ] **Implement KV cache tensors** with `MLShapedArray` and LRU paging.
- [ ] **Provide rope & rotary table kernels** via `MPSGraph` fallback.
- [ ] **Unit test** decoding 20 tokens from a TinyStories checkpoint.

### WS-3 RuntimeLlamaCpp tasks
- [ ] **Create Swift wrapper for llama.cpp C API**
- [ ] **Implement zero-copy bridging for model buffers**
- [ ] **Add chunked decode loop with callback support**
- [ ] **Unit test** parity with CoreML backend on TinyStories
- [ ] **Document building llama.cpp for iOS**


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
        /LlamaCpp
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

## 11. Milestones (suggested 10‑week schedule)  

| Week | Deliverable |
|------|-------------|
| 1 | Phi‑2 converted, TinyStories unit tests passing (WS‑1,4) |
| 2 | CoreMLBackend generates single token, benchmark harness (WS‑2,6) |
| 3 | Streamed generation demo in console |
| 4 | SwiftUI chat UI & model picker (WS‑7) |
| 5 | LlamaCppBackend parity & perf compare (WS‑3) |
| 6 | ModelManager downloads & storage quota (WS‑5) |
| 7 | Energy profiling, KV‑cache paging (WS‑2) |
| 8 | Accessibility & moderation (WS‑7) |
| 9 | Telemetry opt‑in, crash reporting, CI gates (WS‑8,10) |
| 10 | Beta build on TestFlight, dog‑food and iterate |
