## Conversion workflow

This guide explains how to convert a Hugging Face model or GGUF checkpoint into a Core ML `.mlpackage` with INT4 weights.

**Current Status**: The conversion pipeline is fully implemented with support for GPT-2, Phi-2, and Gemma architectures using the ML Program format with stateful models. The Python scripts live under the `Scripts` directory and require Python 3.13.

### 1. Convert the model

Run `convert.py` with a model identifier or local path and an output file path:

```bash
python Scripts/convert.py gpt2 ~/Models/gpt2.mlpackage --seq-length 512
```

By default, the script expects a Hugging Face model identifier. Pass `--gguf`
to convert a local GGUF file instead. The script uses `coremltools` ML Program
format to create stateful models with INT4 quantized weights for optimal
performance on Apple Neural Engine.

### 2. Generate `manifest.json`

After conversion, create a manifest describing the model package. This stores a display name, file size, SHA‑256 hash and the runtime version expected by the app:

```bash
python Scripts/manifest.py ~/Models/gpt2.mlpackage ~/Models/manifest.json --runtime-version 1.0 --name GPT2
```

### 3. Validate perplexity

Use `evaluate_perplexity.py` to ensure the quantized model’s perplexity does not regress by more than 3 % compared to the original:

```bash
python Scripts/evaluate_perplexity.py gpt2 ~/tiny_stories.txt \
    ~/Models/gpt2.mlpackage --max-delta 0.03
```

The CI tests run a similar check on a tiny dataset. If the delta exceeds the threshold, the script exits with a non‑zero status.

### 4. Ship the model

Include the `.mlpackage` and `manifest.json` in your app’s `Application Support` directory or bundle them for TestFlight. The runtime will read the manifest to verify the model before loading it.
