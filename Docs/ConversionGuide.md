## Conversion workflow

This guide explains how to convert a Hugging Face model or gguf checkpoint into a Core ML `.mlpackage` with INT4 weights. The Python scripts live under the `Scripts` directory and require Python 3.13.

### 1. Convert the model

Run `convert.py` with a model identifier or local path and an output file path:

```bash
python Scripts/convert.py gpt2 ~/Models/gpt2.mlpackage --seq-length 512
```

At the moment only PyTorch checkpoints are supported. The script uses `coremltools` to quantize the weights to 4‑bit integers.

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
