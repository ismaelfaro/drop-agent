# LlamaCppBackend: Apple Silicon Metal Acceleration

## Summary

Add a new `LlamaCppBackend` that uses `llama-cpp-python` bindings to run GGUF models directly in-process with Metal GPU acceleration on Apple Silicon Macs. Supports both standard GGUF models (Llama 3, Mistral, Phi) and gpt-oss GGUF models via the existing converter architecture.

## Architecture

```
User CLI: droplet -b llama-cpp -m bartowski/Meta-Llama-3-8B-Instruct-GGUF
                    |
                    v
            LlamaCppBackend(BaseBackend)
            +-- HF download (huggingface_hub)
            +-- Memory detection -> n_gpu_layers
            +-- llama_cpp.Llama(model_path, n_gpu_layers=N, ...)
            +-- generate() -> Ollama-compatible dict
                    |
                    v
          GenerationOrchestrator (unchanged)
```

The backend slots into the existing `BaseBackend` hierarchy alongside `OllamaBackend`, `VLLMBackend`, and `RITSBackend`. No changes to `GenerationOrchestrator`, converters, or tools.

## Backend Class: `LlamaCppBackend`

### Constructor

```python
LlamaCppBackend(
    model_name: str,       # HF repo ID or local .gguf path
    n_gpu_layers: int | None = None,  # None = auto-detect
    n_ctx: int = 8192,
    gguf_file: str | None = None,  # specific file in HF repo
    debug: bool = False,
)
```

### `start()`

Loads the GGUF model into memory:
- Resolves model path (HF download or local file)
- Auto-detects Apple Silicon memory via `sysctl hw.memsize`
- Calculates `n_gpu_layers` if not specified:
  - `available = total_mem * 0.6` (leave room for OS + app)
  - If model fits: `n_gpu_layers = -1` (all layers on GPU)
  - Otherwise: proportional allocation based on model size vs available memory
- Creates `llama_cpp.Llama` instance with Metal flags

### `ensure_model()`

Model resolution:
- If path ends with `.gguf` -> verify local file exists
- Otherwise -> treat as HuggingFace repo ID:
  - Use `huggingface_hub.hf_hub_download()` to fetch GGUF
  - If `gguf_file` specified, download that exact file
  - Otherwise, list repo files and prefer `Q4_K_M` quant, fall back to first `.gguf`
  - Cache location: `~/.cache/huggingface/` (huggingface_hub default)

### `generate(prompt, model, options, timeout)`

Uses `llama.create_completion()` (raw completion, not chat) to match the prompt-based approach used by other backends.

Returns Ollama-compatible dict:
```python
{
    "response": "<generated text>",
    "context": [<prompt_token_ids> + <response_token_ids>],
    "prompt_eval_count": <int>,
}
```

### `stop()`

Deletes the Llama instance to free Metal GPU memory.

## Memory Auto-Detection

```python
import subprocess
result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
total_mem = int(result.stdout.strip())
available = total_mem * 0.6
model_size = os.path.getsize(gguf_path)

if model_size <= available:
    n_gpu_layers = -1  # all on GPU
else:
    ratio = available / model_size
    n_gpu_layers = max(1, int(estimated_total_layers * ratio))
```

Estimated total layers derived from GGUF metadata if available, otherwise use conservative default of 40.

## CLI Integration

### New backend choice

`--backend-type` gains `llama-cpp` option.

### New arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n-gpu-layers` | int | None (auto) | Override GPU layer count for llama.cpp |
| `--n-ctx` | int | 8192 | Context window size for llama.cpp |
| `--gguf-file` | str | None | Specific GGUF filename in HF repo |

### Usage examples

```bash
# Granite 4.0 micro (3B) - fast, good for any Mac
droplet -b llama-cpp -m ibm-granite/granite-4.0-micro-GGUF

# Granite 4.0 tiny (7B) - better quality
droplet -b llama-cpp -m ibm-granite/granite-4.0-tiny-preview-GGUF

# Local GGUF file
droplet -b llama-cpp -m ~/models/granite-4.0-micro.Q4_K_M.gguf

# Override GPU layers
droplet -b llama-cpp -m ibm-granite/granite-4.0-tiny-preview-GGUF --n-gpu-layers 20

# gpt-oss GGUF model (uses Harmony converter automatically via model name detection)
droplet -b llama-cpp -m some-repo/gpt-oss-20b-GGUF
```

## Dependencies

New optional dependency group in `pyproject.toml`:

```toml
[project.optional-dependencies]
metal = [
    "llama-cpp-python",
    "huggingface_hub",
]
```

Install: `pip install droplet[metal]` or `CMAKE_ARGS="-DGGML_METAL=on" pip install droplet[metal]`

## Files to Modify

| File | Change |
|------|--------|
| `droplet/backend.py` | Add `LlamaCppBackend` class |
| `droplet/main.py` | Add `llama-cpp` backend choice + new CLI args |
| `droplet/agent.py` | Add `LlamaCppBackend` import + instantiation branch |
| `pyproject.toml` | Add `[project.optional-dependencies]` metal extra |
| `README.md` | Add llama.cpp backend section |

## Error Handling

- If `llama-cpp-python` not installed: clear error message suggesting `pip install droplet[metal]`
- If GGUF file not found on HF: list available files in the repo
- If model too large for memory: warn and suggest smaller quant or fewer GPU layers
- If not on macOS/Apple Silicon: warn that Metal acceleration unavailable, fall back to CPU
