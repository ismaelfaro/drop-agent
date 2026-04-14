"""Backend server management for Ollama, vLLM, and llama.cpp"""

import os
import platform
import shutil
import subprocess
import time
from abc import ABC, abstractmethod

import requests

from droplet.rich_terminal import blue_print


class BaseBackend(ABC):
    """Abstract base class for LLM backends"""

    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
        self.generate_url = None

    @abstractmethod
    def start(self, timeout=30):
        """Start the backend server if needed"""
        pass

    @abstractmethod
    def stop(self):
        """Stop the backend server if we started it"""
        pass

    @abstractmethod
    def ensure_model(self, model_name):
        """Ensure a model is available"""
        pass

    @abstractmethod
    def generate(self, prompt, model, options, timeout=300):
        """
        Generate completion from the backend

        Args:
            prompt: The prompt string
            model: The model name
            options: Generation options dict
            timeout: Request timeout in seconds

        Returns:
            Response dict with generation results
        """
        pass


class OllamaBackend(BaseBackend):
    """Manages the Ollama server backend"""

    def __init__(self, base_url="http://localhost:11434", debug=False):
        super().__init__(base_url)
        self.host = base_url
        self.generate_url = f"{self.base_url}/api/generate"
        self.process = None
        self.debug = debug

    def is_running(self):
        """Check if Ollama server is already running"""
        response = requests.get(f"{self.host}/api/tags", timeout=5)
        return response.status_code == 200

    def _check_server_running(self):
        """
        Check if server is running, return True/False without raising

        Note: Uses try/except for control flow since connection errors
        are expected when server is not running (not a bug to debug)
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=1)
            return response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False

    def start(self, timeout=30):
        """
        Start the Ollama server if not already running

        Args:
            timeout: Maximum seconds to wait for server to start
        """
        # 1. Check if ollama binary is available
        if not shutil.which("ollama"):
            raise RuntimeError(
                "Ollama binary not found. Please install Ollama first.\n"
                "See README.md for installation and server activation instructions.\n\n"
                "Quick install (macOS): brew install ollama\n"
                "Or download from: https://ollama.com/download"
            )

        # 2. Check if server is already running
        if self._check_server_running():
            return

        # 3. Server not running, start it
        self.process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._check_server_running():
                return
            time.sleep(0.5)

        raise RuntimeError(f"Ollama server failed to start within {timeout} seconds")

    def stop(self):
        """Stop the Ollama server if we started it"""
        if self.process:
            blue_print("🤖 Stopping Ollama server...")
            self.process.terminate()
            self.process.wait()
            self.process = None

    def ensure_model(self, model_name):
        """
        Ensure a model is pulled and ready to use

        Args:
            model_name: Name of the model to pull (e.g., 'llama2')
        """
        response = requests.get(f"{self.host}/api/tags", timeout=10)
        models = response.json()["models"]

        if any(model["name"].startswith(model_name) for model in models):
            return

        blue_print(f"🤖 Pulling model '{model_name}'... (this may take a while)")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull model '{model_name}': {result.stderr}")

        blue_print(f"🤖 Model '{model_name}' pulled successfully")

    def generate(self, prompt, model, options, timeout=300):
        """
        Generate completion using Ollama /api/generate endpoint

        Args:
            prompt: The prompt string
            model: The model name
            options: Generation options dict (temperature, max_tokens, etc.)
            timeout: Request timeout in seconds

        Returns:
            Response dict with 'context', 'prompt_eval_count', and optional 'thinking'
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
            # CRITICAL: Override the model's template to get raw completion mode
            # Without this, Ollama wraps our harmony prompts with its own system message
            # and template, breaking token counting and harmony parsing
            # The "{{ .Prompt }}" template passes our prompt through unchanged
            "template": "{{ .Prompt }}"
            # this wont output the tokens
            # "raw": True
        }

        # Retry logic for Ollama server requests
        max_retries = 3
        retry_delay = 0.1
        result = None

        for retry_attempt in range(max_retries):
            response = requests.post(self.generate_url, json=payload, timeout=timeout)

            if response.status_code != 200:
                error_msg = f"Backend returned status {response.status_code}"
                if self.debug:
                    error_msg += f": {response.text[:200]}"

                if retry_attempt < max_retries - 1:
                    print(f"\033[93m⚠️  {error_msg}. Retrying... (attempt {retry_attempt + 1}/{max_retries})\033[0m")
                    time.sleep(retry_delay)
                    continue

                print(f"\033[91m✗ All {max_retries} attempts failed: {error_msg}\033[0m")
                response.raise_for_status()

            response.raise_for_status()
            result = response.json()
            break

        return result


class VLLMBackend(BaseBackend):
    """Backend for vLLM server (OpenAI-compatible API)"""

    def __init__(self, base_url):
        super().__init__(base_url)
        self.host = base_url
        self.generate_url = f"{self.base_url}/v1/completions"

    def start(self, timeout=30):
        """
        vLLM is expected to be running externally, so this is a no-op
        Just check if the server is accessible
        """
        response = requests.get(f"{self.base_url}/health", timeout=5)
        if response.status_code != 200:
            raise RuntimeError(
                f"vLLM server at {self.base_url} is not accessible. "
                f"Please ensure vLLM server is running at the specified URL."
            )

    def stop(self):
        """vLLM is managed externally, so this is a no-op"""
        pass

    def ensure_model(self, model_name):
        """
        vLLM is expected to have the model loaded, so this is a no-op
        Just verify the model is available
        """
        response = requests.get(f"{self.base_url}/v1/models", timeout=60)
        models_data = response.json()
        available_models = [m["id"] for m in models_data["data"]]

        if model_name not in available_models:
            raise RuntimeError(
                f"Model '{model_name}' not found on vLLM server. "
                f"Available models: {', '.join(available_models)}"
            )

    def generate(self, prompt, model, options, timeout=300):
        """
        Generate completion using vLLM /v1/completions endpoint (OpenAI-compatible)

        Args:
            prompt: The prompt string
            model: The model name
            options: Generation options dict (temperature, max_tokens, etc.)
            timeout: Request timeout in seconds

        Returns:
            Response dict converted to Ollama format with 'context', 'prompt_eval_count'
        """
        # vLLM uses OpenAI-compatible API format
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": options["temperature"],
            "max_tokens": options.get("max_tokens", 32768),
            "stream": False,
            "skip_special_tokens": False,  # Preserve harmony markup tokens
            "return_token_ids": True,  # Return actual token IDs (vLLM v0.10.2+)
        }

        # Add stop tokens if provided
        if "stop_tokens" in options:
            payload["stop_token_ids"] = options["stop_tokens"]

        # Retry logic for vLLM server requests
        max_retries = 3
        retry_delay = 0.1

        for retry_attempt in range(max_retries):
            response = requests.post(self.generate_url, json=payload, timeout=timeout)

            if response.status_code != 200:
                error_msg = f"Backend returned status {response.status_code}: {response.text[:200]}"

                if retry_attempt < max_retries - 1:
                    print(f"\033[93m⚠️  {error_msg}. Retrying... (attempt {retry_attempt + 1}/{max_retries})\033[0m")
                    time.sleep(retry_delay)
                    continue

                print(f"\033[91m✗ All {max_retries} attempts failed: {error_msg}\033[0m")
                response.raise_for_status()

            break

        result = response.json()

        choice = result["choices"][0]

        # Convert vLLM response format to Ollama-compatible format
        # With return_token_ids=True, vLLM returns actual token IDs
        # vLLM returns: {"prompt_token_ids": [...], "choices": [{"text": ..., "token_ids": [...]}]}
        # Ollama expects: {"context": [...], "prompt_eval_count": N, "response": "..."}

        response_text = choice["text"]

        # Get actual token IDs from vLLM (requires vLLM v0.10.2+)
        # Both are in the choice object
        prompt_token_ids = choice["prompt_token_ids"]
        response_token_ids = choice["token_ids"]

        # Build context as concatenation of prompt and response tokens
        context_tokens = prompt_token_ids + response_token_ids

        # Return in Ollama-compatible format
        return {
            "response": response_text,
            "context": context_tokens,
            "prompt_eval_count": len(prompt_token_ids),
        }


class RITSBackend(VLLMBackend):
    """Backend for RITS (vLLM-based API with per-model endpoints)"""

    def __init__(self, base_url, api_key):
        # Don't call super().__init__ yet - we need to fetch the model endpoint first
        self.api_key = api_key
        self.rits_info_url = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
        self.model_endpoints = None
        # Placeholder until we fetch the actual endpoint
        self.base_url = base_url
        self.host = base_url

    def _fetch_available_models(self):
        """Fetch list of available models from RITS API"""
        from collections import Counter
        from urllib.parse import urlparse

        headers = {"RITS_API_KEY": self.api_key}
        response = requests.get(self.rits_info_url, headers=headers, timeout=10)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch RITS model list (status {response.status_code}). "
                f"Please check your RITS API key.\n{response.text}"
            )

        # Store raw model data for listing purposes
        model_data = response.json()
        self.raw_model_data = model_data

        # Count occurrences of each model name to find duplicates
        name_counts = Counter(m["model_name"] for m in model_data)

        # Build map of model_name -> endpoint, filtering out invalid mappings
        self.model_endpoints = {}
        for model in model_data:
            model_name = model["model_name"]
            endpoint = model["endpoint"]

            # If name is unique, always include it
            if name_counts[model_name] == 1:
                self.model_endpoints[model_name] = f"{endpoint}/v1"
            else:
                # Name appears multiple times - only use if basename matches
                parsed_url = urlparse(endpoint)
                basename = parsed_url.path.strip('/').split('/')[-1]

                # Check if basename matches the model name (or last part after /)
                model_basename = model_name.split('/')[-1]
                if basename == model_basename:
                    self.model_endpoints[model_name] = f"{endpoint}/v1"

        return self.model_endpoints

    def _suggest_similar_models(self, requested_model, available_models):
        """Suggest similar model names in case of typo"""
        from difflib import get_close_matches

        suggestions = get_close_matches(requested_model, available_models, n=3, cutoff=0.6)
        return suggestions

    def start(self, timeout=30):
        """
        Fetch available models from RITS - no server to start

        Note: Uses try/except for control flow since connection errors
        are expected when RITS is unreachable (not a bug to debug)
        """
        try:
            self._fetch_available_models()
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            raise RuntimeError(
                "Failed to connect to RITS server.\n"
                "Connection to rits.fmaas.res.ibm.com timed out.\n"
                "Please check your network connection or VPN."
            )

    def stop(self):
        """RITS is managed externally, so this is a no-op"""
        pass

    def ensure_model(self, model_name):
        """
        Verify the model is available on RITS and set the correct endpoint
        """
        if self.model_endpoints is None:
            self._fetch_available_models()

        if model_name not in self.model_endpoints:
            # Model not found - suggest similar ones
            suggestions = self._suggest_similar_models(model_name, list(self.model_endpoints.keys()))

            error_msg = f"Model '{model_name}' not found on RITS."

            if suggestions:
                error_msg += "\n\nDid you mean one of these?\n  • " + "\n  • ".join(suggestions)
            else:
                available = list(self.model_endpoints.keys())
                error_msg += f"\n\nAvailable models: {', '.join(available[:5])}"
                if len(available) > 5:
                    error_msg += f", ... ({len(available)} total)"

            raise RuntimeError(error_msg)

        # Set the base_url to the model-specific endpoint
        self.base_url = self.model_endpoints[model_name]
        self.host = self.base_url
        self.generate_url = f"{self.base_url}/completions"

        # Verify the endpoint is accessible
        # Health endpoint is at base without /v1
        base_endpoint = self.base_url.rstrip('/v1')
        headers = {"RITS_API_KEY": self.api_key}
        health_url = f"{base_endpoint}/health"
        response = requests.get(health_url, headers=headers, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"RITS endpoint for model '{model_name}' is not accessible.\n"
                f"Endpoint: {self.base_url}\n"
                f"Health check returned status {response.status_code}"
            )

    def generate(self, prompt, model, options, timeout=300):
        """
        Generate completion using RITS /v1/completions endpoint (OpenAI-compatible)
        Adds RITS_API_KEY header to requests
        """
        # vLLM uses OpenAI-compatible API format
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": options["temperature"],
            "max_tokens": options.get("max_tokens", 32768),
            "stream": False,
            "skip_special_tokens": False,
            "return_token_ids": True,
        }

        # Add stop tokens if provided
        if "stop_tokens" in options:
            payload["stop_token_ids"] = options["stop_tokens"]

        # Add RITS API key header
        headers = {"RITS_API_KEY": self.api_key}

        response = requests.post(self.generate_url, json=payload, headers=headers, timeout=timeout)

        if response.status_code != 200:
            error_msg = f"RITS returned status {response.status_code}: {response.text}"
            print(f"\033[91mERROR: {error_msg}\033[0m")
            response.raise_for_status()

        result = response.json()
        choice = result["choices"][0]

        response_text = choice["text"]
        prompt_token_ids = choice["prompt_token_ids"]
        response_token_ids = choice["token_ids"]

        context_tokens = prompt_token_ids + response_token_ids

        return {
            "response": response_text,
            "context": context_tokens,
            "prompt_eval_count": len(prompt_token_ids),
        }


class LlamaCppBackend(BaseBackend):
    """Backend using llama-cpp-python bindings with Metal GPU acceleration for Apple Silicon"""

    def __init__(self, model_name, n_gpu_layers=None, n_ctx=8192, gguf_file=None, debug=False):
        super().__init__("local://llama.cpp")
        self.model_name = model_name
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.gguf_file = gguf_file
        self.debug = debug
        self.llm = None
        self.model_path = None

    def _detect_gpu_layers(self, model_path):
        """Auto-detect optimal n_gpu_layers based on available memory on Apple Silicon"""
        if platform.system() != "Darwin":
            blue_print("Not on macOS — Metal acceleration unavailable, using CPU only")
            return 0

        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            total_mem = int(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError):
            blue_print("Could not detect system memory, defaulting to all layers on GPU")
            return -1

        available = total_mem * 0.6
        model_size = os.path.getsize(model_path)

        if model_size <= available:
            return -1  # all layers on GPU

        # Estimate total layers from GGUF metadata if possible, otherwise default to 40
        estimated_layers = self._estimate_total_layers(model_path)
        ratio = available / model_size
        n_layers = max(1, int(estimated_layers * ratio))
        blue_print(f"Model ({model_size / 1e9:.1f}GB) exceeds 60% of RAM ({total_mem / 1e9:.1f}GB) — offloading {n_layers}/{estimated_layers} layers to GPU")
        return n_layers

    def _estimate_total_layers(self, model_path):
        """Estimate total layer count from GGUF metadata"""
        try:
            from llama_cpp import Llama
            metadata = Llama.metadata(model_path)
            for key in metadata:
                if "block_count" in key:
                    return int(metadata[key]) + 1  # +1 for output layer
        except Exception:
            pass
        return 40  # conservative default

    def _resolve_model_path(self):
        """Resolve model name to a local GGUF file path, downloading from HF if needed"""
        expanded = os.path.expanduser(self.model_name)
        if expanded.endswith(".gguf") or os.path.isfile(expanded):
            if not os.path.isfile(expanded):
                raise RuntimeError(f"GGUF file not found: {expanded}")
            return expanded

        # HuggingFace repo ID
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required to download models from HuggingFace.\n"
                "Install with: pip install droplet[metal]"
            )

        repo_id = self.model_name
        blue_print(f"Resolving GGUF model from HuggingFace: {repo_id}")

        if self.gguf_file:
            filename = self.gguf_file
        else:
            try:
                files = list_repo_files(repo_id)
            except Exception as e:
                raise RuntimeError(f"Failed to list files in HuggingFace repo '{repo_id}': {e}")

            gguf_files = [f for f in files if f.endswith(".gguf")]
            if not gguf_files:
                raise RuntimeError(f"No .gguf files found in repo '{repo_id}'")

            # Prefer Q4_K_M quant
            preferred = [f for f in gguf_files if "Q4_K_M" in f]
            filename = preferred[0] if preferred else gguf_files[0]

        # Check if already cached to skip download progress
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        except Exception:
            cached = None

        if cached and isinstance(cached, str):
            blue_print(f"Using cached model: {cached}")
            return cached

        # Download with rich progress bar
        import threading

        from rich.progress import (BarColumn, DownloadColumn, Progress,
                                   TextColumn, TimeRemainingColumn,
                                   TransferSpeedColumn)

        # Get file size from HF metadata for accurate progress
        total_size = None
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
            url = hf_hub_url(repo_id=repo_id, filename=filename)
            metadata = get_hf_file_metadata(url)
            total_size = metadata.size
        except Exception:
            pass

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )

        # Suppress hf_hub's own tqdm bars
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        download_result = [None]
        download_error = [None]

        def _download():
            try:
                download_result[0] = hf_hub_download(repo_id=repo_id, filename=filename)
            except Exception as e:
                download_error[0] = e

        thread = threading.Thread(target=_download)

        try:
            with progress:
                task_id = progress.add_task(filename, total=total_size)
                thread.start()

                # Poll the incomplete download file for progress
                from huggingface_hub.constants import HF_HUB_CACHE
                blob_dir = os.path.join(
                    HF_HUB_CACHE,
                    f"models--{repo_id.replace('/', '--')}",
                    "blobs",
                )

                while thread.is_alive():
                    # Look for .incomplete files being written
                    if os.path.isdir(blob_dir):
                        for f in os.listdir(blob_dir):
                            if f.endswith(".incomplete"):
                                fpath = os.path.join(blob_dir, f)
                                try:
                                    current = os.path.getsize(fpath)
                                    progress.update(task_id, completed=current)
                                except OSError:
                                    pass
                                break
                    thread.join(timeout=0.3)

                thread.join()

                if download_error[0]:
                    raise download_error[0]

                local_path = download_result[0]

                # Mark complete
                file_size = os.path.getsize(local_path)
                progress.update(task_id, total=file_size, completed=file_size)

        except Exception as e:
            raise RuntimeError(f"Failed to download {filename} from '{repo_id}': {e}")
        finally:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

        model_size = os.path.getsize(local_path)
        blue_print(f"Model downloaded ({model_size / 1e9:.1f}GB): {local_path}")
        return local_path

    def start(self, timeout=30):
        """Load the GGUF model into memory with Metal acceleration"""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is required for the llama-cpp backend.\n"
                "Install with: CMAKE_ARGS=\"-DGGML_METAL=on\" pip install droplet[metal]"
            )

        self.model_path = self._resolve_model_path()

        if self.n_gpu_layers is not None:
            n_gpu_layers = self.n_gpu_layers
        else:
            n_gpu_layers = self._detect_gpu_layers(self.model_path)

        from rich.console import Console
        from rich.live import Live
        from rich.spinner import Spinner

        blue_print(f"Loading model with n_gpu_layers={n_gpu_layers}, n_ctx={self.n_ctx}")

        spinner = Spinner("dots", text="[bold blue]Loading model into memory (Metal GPU)...")
        console = Console()

        with Live(spinner, console=console, refresh_per_second=10):
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=self.debug,
            )

        blue_print("Model loaded and ready")

    def stop(self):
        """Release the model from memory"""
        if self.llm is not None:
            del self.llm
            self.llm = None

    def ensure_model(self, model_name):
        """Model is already loaded in start(), nothing to do"""
        if self.llm is None:
            raise RuntimeError("LlamaCpp model not loaded. Call start() first.")

    def generate(self, prompt, model, options, timeout=300):
        """
        Generate completion using llama-cpp-python

        Returns Ollama-compatible dict with 'response', 'context', and 'prompt_eval_count'
        """
        if self.llm is None:
            raise RuntimeError("LlamaCpp model not loaded. Call start() first.")

        prompt_tokens = self.llm.tokenize(prompt.encode("utf-8"))

        result = self.llm.create_completion(
            prompt,
            max_tokens=options.get("max_tokens", 32768),
            temperature=options.get("temperature", 0.0),
            stop=None,
            echo=False,
        )

        response_text = result["choices"][0]["text"]
        response_tokens = self.llm.tokenize(response_text.encode("utf-8"))
        context_tokens = prompt_tokens + response_tokens

        return {
            "response": response_text,
            "context": context_tokens,
            "prompt_eval_count": len(prompt_tokens),
        }
