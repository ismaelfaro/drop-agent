# Backend Configuration

DROP supports four backends for LLM inference. Each backend implements the same interface, so all agent features (tools, context compaction, etc.) work identically regardless of backend.

## Ollama

Local inference server. Good default for laptop usage.

### Setup

    brew install ollama
    brew services start ollama

### Usage

    droplet -b ollama -m gpt-oss:20b
    droplet -b ollama -m gpt-oss:20b -u http://localhost:11434

Ollama is the default backend. If the server isn't running, droplet will attempt to start it automatically.

### Pulling models

Models are pulled automatically on first use. To pull manually:

    ollama pull gpt-oss:20b

## vLLM

Remote inference via OpenAI-compatible API. Use when you have access to a GPU server running vLLM.

### Setup

If you need to launch vLLM yourself:

    vllm serve $MODEL --host 0.0.0.0 --port $PORT

### Usage

    droplet -b vllm -m $MODEL -u http://${HOSTNAME}:${PORT}

### Saving configs

Store connection details for reuse:

    droplet -b vllm -m $MODEL -u http://${HOSTNAME}:${PORT} --save-config remote-vllm
    droplet -c remote-vllm

Any additional arguments override the saved config values.

## RITS

IBM RITS service (vLLM-based with per-model endpoints). Requires a RITS API key.

### Usage

    droplet -b rits-vllm -m $MODEL --rits-api-key $KEY

The API key can also be set via the `RITS_API_KEY` environment variable.

### Listing available models

    droplet --rits-list-models --rits-api-key $KEY

## llama.cpp (Apple Silicon)

Direct in-process inference using llama-cpp-python with Metal GPU acceleration. Best performance on Mac M1/M2/M3/M4. No external server needed.

### Install

    CMAKE_ARGS="-DGGML_METAL=on" pip install droplet[metal]

This installs `llama-cpp-python` (compiled with Metal support) and `huggingface_hub` for model downloads.

### Usage

Auto-download an IBM Granite model from HuggingFace:

    # 3B model - fast, runs on any Mac
    droplet -b llama-cpp -m ibm-granite/granite-4.0-micro-GGUF

    # 7B model - better quality
    droplet -b llama-cpp -m ibm-granite/granite-4.0-tiny-preview-GGUF

Use a local GGUF file:

    droplet -b llama-cpp -m ~/models/granite-4.0-micro.Q4_K_M.gguf

Specify a particular quantization from a HuggingFace repo:

    droplet -b llama-cpp -m ibm-granite/granite-4.0-tiny-preview-GGUF --gguf-file granite-4.0-tiny-preview-Q5_K_M.gguf

### GPU layer allocation

By default, droplet auto-detects available unified memory and offloads as many layers as will fit to the Metal GPU. Override manually:

    droplet -b llama-cpp -m ibm-granite/granite-4.0-tiny-preview-GGUF --n-gpu-layers 20

Set `--n-gpu-layers 0` to force CPU-only inference.

### Context window

Default context window is 8192 tokens. Adjust with `--n-ctx`:

    droplet -b llama-cpp -m ibm-granite/granite-4.0-micro-GGUF --n-ctx 4096

### Recommended GGUF models

#### IBM Granite 4.0

| Model | Size | Repo ID |
|-------|------|---------|
| Granite 4.0 Micro | 3B | `ibm-granite/granite-4.0-micro-GGUF` |
| Granite 4.0 Tiny | 7B | `ibm-granite/granite-4.0-tiny-preview-GGUF` |
| Granite 4.0 H-Micro | 3B | `ibm-granite/granite-4.0-h-micro-GGUF` |
| Granite 4.0 H-Small | 32B | `ibm-granite/granite-4.0-h-small-base-GGUF` |

#### Google Gemma 4

| Model | Size | Repo ID |
|-------|------|---------|
| Gemma 4 E2B | 5B | `ggml-org/gemma-4-E2B-it-GGUF` |
| Gemma 4 E4B | 8B | `ggml-org/gemma-4-E4B-it-GGUF` |
| Gemma 4 27B (MoE) | 25B | `ggml-org/gemma-4-26B-A4B-it-GGUF` |
| Gemma 4 31B | 31B | `ggml-org/gemma-4-31B-it-GGUF` |

Example:

    droplet -b llama-cpp -m ibm-granite/granite-4.0-micro-GGUF

### Non-macOS platforms

On Linux/Windows, `llama-cpp-python` falls back to CPU inference. For GPU acceleration on those platforms, use the Ollama or vLLM backends instead.
