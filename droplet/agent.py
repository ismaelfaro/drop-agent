"""Agent scaffold using completions API with Harmony message conversion"""

import asyncio
import datetime
import json
import logging
import os
import time

from openai_harmony import (Author, HarmonyError, Message, ReasoningEffort,
                            Role, SystemContent, TextContent)

from droplet.backend import LlamaCppBackend, OllamaBackend, RITSBackend, VLLMBackend
from droplet.converters import get_converter_for_model
from droplet.generation_orchestrator import (Backend500Error,
                                             BackendConnectionError,
                                             BackendHTTPError, GenerationError,
                                             GenerationOrchestrator)
from droplet.rich_terminal import blue_print, debug_print_error

# Configure logging to reduce verbosity from gpt_oss and related libraries
# gpt_oss uses structlog, so we need to configure it
try:
    import structlog

    # Get the current configuration
    processors = structlog.get_config().get("processors", [])

    # Add a filter to drop warning/error logs from gpt_oss components
    def filter_gpt_oss_logs(logger, method_name, event_dict):
        # Drop logs from gpt_oss components
        if "component" in event_dict:
            component = event_dict.get("component", "")
            if component.startswith("gpt_oss"):
                raise structlog.DropEvent
        return event_dict

    # Reconfigure structlog with our filter
    structlog.configure(
        processors=[filter_gpt_oss_logs] + processors,
    )
except ImportError:
    pass

# Also configure standard logging as fallback
for logger_name in [
    'gpt_oss',
    'gpt_oss.tools',
    'gpt_oss.tools.simple_browser',
    'gpt_oss.tools.simple_browser.simple_browser_tool',
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False


class DropletAgent:
    """
    Model-independent agent scaffold using pluggable backends and message converters

    This implementation uses abstract message converters to support multiple model families:
    - GPT-OSS models: HarmonyMessageConverter with harmony encoding/decoding
    - Granite/HF models: GraniteMessageConverter with transformers chat templates

    Architecture:
    - MessageConverter abstraction handles model-specific formatting
    - Backend layer (Ollama/vLLM/RITS) provides model-agnostic inference
    - Tool execution uses harmony Message objects internally
    - Converter translates between Message objects and model-specific formats

    Supported Backends:
    - Ollama: Local backend using /api/generate endpoint
    - vLLM: Remote backend using OpenAI-compatible /v1/completions endpoint
    - RITS: Remote RITS backend with API key authentication

    Key Design:
    - Converter selection based on model name pattern matching
    - Tools use ToolNamespaceConfig, converter translates to model format
    - Response parsing handles both token-based (Harmony) and text-based (Granite)
    - Tool calls detected via 'recipient' field in parsed messages

    See droplet.converters module for converter implementations.
    """

    # Class-level prompt constants
    LOOP_TOOL_FAIL = (
        "I couldn't find what you were looking for after several attempts. Can you rephrase your question or provide "
        "more details about what you need?"
    )

    def __init__(
        self,
        model="gpt-oss:20b",
        base_url="http://localhost:11434",
        backend_type="ollama",
        rits_api_key=None,
        debug=False,
        # io
        log_file=None,
        out_messages=None,
        # tool config
        restricted_tools=None,
        tool_names=None,
        # milvus tools
        milvus_db=None,
        milvus_model=None,
        milvus_collection=None,
        # bcp tool
        bcp_server_url=None,
        # semantic scholar
        semantic_scholar_api_key=None,
        # prompts
        no_droplet_sytem_prompt=False,
        system_prompt=None,
        developer_prompt=None,
        loop_tool_fail=None,
        input_prefix=None,
        gpt_reasoning=None,
        # generation
        temperature=0.0,
        max_tokens=32768,
        max_iterations=10,
        # llama.cpp
        n_gpu_layers=None,
        n_ctx=8192,
        gguf_file=None,
        # context compaction
        context_compaction_method=None,
        context_compaction_threshold=64000,
        max_context_compactions=3,
        compaction_keep_n=5,
    ):
        """
        Initialize the agent

        Args:
            model: Model name (default: gpt-oss:20b)
            base_url: Base URL for backend API (default: http://localhost:11434)
            backend_type: Backend type - "ollama", "vllm", or "rits-vllm" (default: ollama)
            debug: Enable debug mode (default: False)
            restricted_tools: Set of tool class names that require user permission
            tool_names: List of tool class names to load (default: ['FileBrowserTool'])
            milvus_db: Path to Milvus database file (required for RetrieverBrowserTool)
            milvus_model: SentenceTransformer model name or path
            milvus_collection: Milvus collection name
            log_file: Path to JSON file to log conversation history
            out_messages: Path to JSON file to write final message list (for batch evaluation)
            bcp_server_url: BCP search server URL (required for BCPBrowserTool)
            rits_api_key: RITS API key (required for rits-vllm backend)
            semantic_scholar_api_key: Semantic Scholar API key (optional, for higher rate limits)
            no_droplet_sytem_prompt: Disable default Droplet system prompt (default: False)
            system_prompt: Override system prompt (used if no_droplet_sytem_prompt=True)
            developer_prompt: Additional developer instructions added as developer message
            loop_tool_fail: Override default loop failure message (default: class-level LOOP_TOOL_FAIL)
            input_prefix: Prefix to add to user input messages (e.g., "Question: ") (default: None)
            gpt_reasoning: GPT-OSS reasoning effort level: "low", "medium", or "high" (default: None, uses model default)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum tokens to generate (default: 32768)
            max_iterations: Maximum tool call iterations per user input (default: 10)
            context_compaction_method: Method name ("keep_last_n", "llm_keep_last_n", or "llm"); None disables
            context_compaction_threshold: Token count threshold to trigger compaction (default: 64000)
            max_context_compactions: Max times compaction can fire per user_input() call (default: 3)
            compaction_keep_n: Number of recent messages to keep after compaction, must be >= 1 (default: 5)
        """
        # Initialize backend based on type
        if backend_type == "ollama":
            self.backend = OllamaBackend(base_url=base_url, debug=debug)
        elif backend_type == "vllm":
            self.backend = VLLMBackend(base_url=base_url)
        elif backend_type == "rits-vllm":
            if not rits_api_key:
                raise RuntimeError("RITS API key is required for rits-vllm backend. Use --rits-api-key argument.")
            self.backend = RITSBackend(base_url=base_url, api_key=rits_api_key)
        elif backend_type == "llama-cpp":
            self.backend = LlamaCppBackend(
                model_name=model,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                gguf_file=gguf_file,
                debug=debug,
            )
        else:
            raise RuntimeError(f"Unknown backend type: {backend_type}. Must be 'ollama', 'vllm', 'rits-vllm', or 'llama-cpp'")

        self.backend.start()
        self.backend.ensure_model(model)

        self.model = model
        self.debug = debug
        self.base_url = base_url.rstrip('/')

        # Initialize message converter for this model
        self.converter = get_converter_for_model(self.model)

        # Initialize generation orchestrator
        self.orchestrator = GenerationOrchestrator(self.backend, self.converter)

        # Setup tokenizer and context limits via converter
        self._setup_tokenizer_and_limits()

        # generator options
        self.options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop_tokens": self.converter.get_stop_tokens(),
        }
        self.max_iterations = max_iterations

        # Context compaction settings
        self.context_compaction_method = context_compaction_method
        self.context_compaction_threshold = context_compaction_threshold
        self.max_context_compactions = max_context_compactions
        if compaction_keep_n < 1:
            raise ValueError(f"compaction_keep_n must be >= 1, got {compaction_keep_n}")
        self.compaction_keep_n = compaction_keep_n

        # Initialize state
        self.state = {}

        # Tool permission tracking
        self.restricted_tools = restricted_tools or set()
        self.allowed_tools = set()  # Tools allowed for this session

        # Log file for conversation history
        if log_file and not log_file.endswith('.json'):
            raise RuntimeError(f"Log file must end with .json, got: {log_file}")
        self.log_file = log_file

        # Output messages file for batch evaluation
        if out_messages and not out_messages.endswith('.json'):
            raise RuntimeError(f"Output messages file must end with .json, got: {out_messages}")
        self.out_messages = out_messages

        # Store prompts and reasoning level for later use
        self.developer_prompt = developer_prompt
        self.gpt_reasoning = gpt_reasoning

        # Setup prompts before initializing tools (needed by _setup_tool_messages)
        if not no_droplet_sytem_prompt:
            # Use converter's model-specific system prompt
            self.SYSTEM_PROMPT = self.converter.get_default_system_prompt(self.model)
        else:
            self.SYSTEM_PROMPT = system_prompt

        # Set instance-level prompts (use class defaults if not provided)
        if loop_tool_fail is not None:
            self.LOOP_TOOL_FAIL = loop_tool_fail
        else:
            self.LOOP_TOOL_FAIL = DropletAgent.LOOP_TOOL_FAIL

        # Store input prefix (default: no prefix)
        self.input_prefix = input_prefix if input_prefix is not None else ""

        # Initialize tools
        tools = self._initialize_tools(tool_names, milvus_db, milvus_model, milvus_collection, bcp_server_url, semantic_scholar_api_key)

        # Setup initial conversation with system messages and tools
        initial_messages, self.tool_instances = self._setup_tool_messages(tools)
        self.conversation_history = initial_messages

        # Track how many prefix messages (SYSTEM + optional DEVELOPER) to always preserve during compaction
        self._num_system_messages = len(initial_messages)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup backend"""
        self.backend.stop()
        return False

    def _save_conversation_log(self):
        """Save conversation history to JSON file"""
        if not self.log_file:
            return

        # Convert Message objects to dicts
        conversation_dicts = []
        for msg in self.conversation_history:
            if hasattr(msg, 'to_dict'):
                conversation_dicts.append(msg.to_dict())
            else:
                # Fallback if to_dict not available
                conversation_dicts.append(msg)

        # Write to file with indentation
        with open(self.log_file, 'w') as f:
            json.dump(conversation_dicts, f, indent=2)

    def _save_out_messages(self):
        """Save final message list to JSON file for batch evaluation"""
        if not self.out_messages:
            return

        # Convert Message objects to dicts
        message_dicts = []
        for msg in self.conversation_history:
            if hasattr(msg, 'to_dict'):
                message_dicts.append(msg.to_dict())
            else:
                # Fallback if to_dict not available
                message_dicts.append(msg)

        # Write to file with indentation
        with open(self.out_messages, 'w') as f:
            json.dump(message_dicts, f, indent=2)

    def _initialize_tools(self, tool_names, milvus_db, milvus_model, milvus_collection, bcp_server_url, semantic_scholar_api_key):
        """
        Initialize tool instances from tool names

        Args:
            tool_names: List of tool class names to load
            milvus_db: Path to Milvus database file
            milvus_model: SentenceTransformer model name
            milvus_collection: Milvus collection name
            bcp_server_url: BCP search server URL
            semantic_scholar_api_key: Semantic Scholar API key

        Returns:
            List of tool instances
        """
        import inspect

        from droplet import tools as tools_module

        # Use default tools if none specified
        if tool_names is None:
            tool_names = ['FileBrowserTool']

        # Discover all *Tool classes from droplet.tools
        # Exclude base classes
        excluded_tools = {'SimpleFunctionTool'}
        available_tools = {}
        for name, obj in inspect.getmembers(tools_module, inspect.isclass):
            if name.endswith('Tool') and name not in excluded_tools:
                if hasattr(obj, '__module__') and obj.__module__.startswith('droplet'):
                    available_tools[name] = obj

        # Initialize tool instances
        tools = []
        for tool_name in tool_names:
            if tool_name not in available_tools:
                raise RuntimeError(
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {', '.join(sorted(available_tools.keys()))}"
                )

            tool_class = available_tools[tool_name]

            # Special handling for RetrieverBrowserTool which requires Milvus configuration
            if tool_name == 'RetrieverBrowserTool':
                if not milvus_db:
                    raise RuntimeError(
                        "RetrieverBrowserTool requires milvus_db argument"
                    )

                tools.append(tool_class(
                    milvus_db=milvus_db,
                    milvus_model=milvus_model,
                    milvus_collection=milvus_collection
                ))
            # BCPBrowserTool requires BCP server URL
            elif tool_name == 'BCPBrowserTool':
                if not bcp_server_url:
                    raise RuntimeError(
                        "BCPBrowserTool requires bcp_server_url argument (use --bcp-server-url)"
                    )
                tools.append(tool_class(base_url=bcp_server_url))
            # SemanticScholarTool accepts optional API key
            elif tool_name == 'SemanticScholarTool':
                tools.append(tool_class(api_key=semantic_scholar_api_key))
            else:
                tools.append(tool_class())

        if not tools:
            raise RuntimeError("No tools were loaded")

        return tools

    def _setup_tokenizer_and_limits(self):
        """Setup tokenizer and context limits via converter"""
        self.max_context_tokens = self.converter.get_max_context_tokens()

    def _format_tokens(self, tokens):
        """Format token count as thousands (K)"""
        if tokens >= 1000:
            result = f"{tokens / 1000:.1f}K"
        else:
            result = str(tokens)
        return result

    def _ask_tool_permission(self, tool_class_name, function_name, function_args):
        """Ask user for permission to execute a restricted tool"""
        # Check if already allowed for this session (check by tool class, not function)
        permission_granted = tool_class_name in self.allowed_tools

        if not permission_granted:
            print("\n \033[93m⚠️\033[0m  \033[94mRestricted Tool Request\033[0m")

            # Special handling for PythonTool - pretty print the code
            if tool_class_name == "PythonTool":
                # Extract the script from function_args
                script = function_args.get('script', '')

                print(f" \033[94mTool: {tool_class_name}.{function_name}\033[0m")
                print("\n \033[94mPython code to execute:\033[0m")

                # Pretty print Python code using rich
                from rich.console import Console
                from rich.syntax import Syntax

                console = Console()
                syntax = Syntax(script, "python", theme="monokai", line_numbers=True)
                console.print(syntax)
            else:
                # Format arguments for display (non-Python tools)
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in function_args.items())
                print(f" \033[94mTool: {function_name}({args_str})\033[0m")

            print("\n \033[94mOptions:\033[0m")
            print(" \033[94m  [1] Yes, execute once\033[0m")
            print(f" \033[94m  [2] Yes, allow all calls to {tool_class_name} this session\033[0m")
            print(" \033[94m  [3] No, cancel\033[0m")

            valid_choice = False
            while not valid_choice:
                choice = input(" \033[94mYour choice:\033[0m ").strip()

                if choice == '1':
                    permission_granted = True
                    valid_choice = True
                elif choice == '2':
                    self.allowed_tools.add(tool_class_name)
                    print(f" \033[94m✓ All '{tool_class_name}' calls allowed for this session\033[0m")
                    permission_granted = True
                    valid_choice = True
                elif choice == '3':
                    print(" \033[94m✗ Tool call cancelled\033[0m")
                    permission_granted = False
                    valid_choice = True
                else:
                    print(" \033[94mInvalid choice. Please enter 1, 2, or 3\033[0m")

        return permission_granted

    def _setup_tool_messages(self, tools):
        """Setup initial messages with tool configuration"""
        tool_instances = {}

        if tools:
            # Register each tool with its own namespace (files.search, wiki.search, etc.)
            # This matches gpt-oss SimpleBrowserTool behavior

            tool_configs = []
            for t in tools:
                tool_config = t.tool_config
                tool_configs.append(tool_config)

                # Register tool instance for each function in its namespace
                namespace_name = tool_config.name
                # Convert to dict to access tool definitions
                tool_config_dict = tool_config.model_dump()
                tool_defs = tool_config_dict.get('tools', [])

                if len(tool_defs) == 0:
                    # Direct-content tool (no functions) - register by namespace name only
                    # E.g., "python" for PythonTool
                    tool_instances[namespace_name] = t
                    tool_instances[f"functions.{namespace_name}"] = t
                else:
                    # Function-based tool - register each function
                    for tool_def in tool_defs:
                        # Format: "namespace.function" (e.g., "files.search", "wiki.open")
                        function_name = f"{namespace_name}.{tool_def['name']}"
                        tool_instances[function_name] = t
                        # Also register with "functions." prefix since Harmony may add it
                        tool_instances[f"functions.{function_name}"] = t
                        # FIXME: Also register with "browser." prefix since model may use that for search tools
                        tool_instances[f"browser.{function_name}"] = t

            # Create SystemContent with multiple tool namespaces
            system_content = SystemContent.new()

            # Set reasoning effort if specified
            if self.gpt_reasoning:
                reasoning_map = {
                    "low": ReasoningEffort.LOW,
                    "medium": ReasoningEffort.MEDIUM,
                    "high": ReasoningEffort.HIGH,
                }
                if self.gpt_reasoning.lower() in reasoning_map:
                    system_content = system_content.with_reasoning_effort(reasoning_map[self.gpt_reasoning.lower()])

            # Set conversation start date (important for proper harmony formatting)
            system_content = system_content.with_conversation_start_date(
                datetime.datetime.now().strftime("%Y-%m-%d")
            )

            for config in tool_configs:
                system_content = system_content.with_tools(config)

            # Only override system prompt if explicitly provided (keeps default ChatGPT identity otherwise)
            if self.SYSTEM_PROMPT is not None:
                system_content.model_identity = self.SYSTEM_PROMPT

            # DON'T set model_identity when using tools - the tools configuration handles harmony format
            # If custom prompt needed, use developer_prompt instead
            system_message = Message.from_role_and_content(Role.SYSTEM, system_content)

            messages = [system_message]

            # Add developer prompt as an additional developer message if provided
            if self.developer_prompt:
                developer_message = Message.from_role_and_content(
                    Role.DEVELOPER,
                    TextContent(text=self.developer_prompt)
                )
                messages.append(developer_message)
        else:
            # No tools - create SystemContent for reasoning effort
            system_content = SystemContent.new()

            # Set reasoning effort if specified
            if self.gpt_reasoning:
                reasoning_map = {
                    "low": ReasoningEffort.LOW,
                    "medium": ReasoningEffort.MEDIUM,
                    "high": ReasoningEffort.HIGH,
                }
                if self.gpt_reasoning.lower() in reasoning_map:
                    system_content = system_content.with_reasoning_effort(reasoning_map[self.gpt_reasoning.lower()])

            # Set conversation start date (important for proper harmony formatting)
            system_content = system_content.with_conversation_start_date(
                datetime.datetime.now().strftime("%Y-%m-%d")
            )

            # Only override system prompt if explicitly provided (keeps default ChatGPT identity otherwise)
            if self.SYSTEM_PROMPT is not None:
                system_content.model_identity = self.SYSTEM_PROMPT

            system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
            messages = [system_message]

            # Add developer prompt as an additional developer message if provided
            if self.developer_prompt:
                developer_message = Message.from_role_and_content(
                    Role.DEVELOPER,
                    TextContent(text=self.developer_prompt)
                )
                messages.append(developer_message)

        return messages, tool_instances

    def _replace_cursor_with_url_in_display_args(self, recipient, display_args, tool_instances):
        """Replace cursor argument with URL/path for browser tools in display args"""
        if recipient not in tool_instances:
            return
        if "cursor" not in display_args:
            return

        tool = tool_instances[recipient]
        if not hasattr(tool, 'tool_state'):
            return
        if not hasattr(tool.tool_state, 'page_stack'):
            return

        cursor = display_args["cursor"]
        page_stack = tool.tool_state.page_stack

        url = None
        if isinstance(cursor, int):
            if cursor == -1:
                if len(page_stack) > 0:
                    url = page_stack[-1]
            else:
                if 0 <= cursor < len(page_stack):
                    url = page_stack[cursor]

        if url is None:
            return

        display_path = url
        if url.startswith("file://"):
            from urllib.parse import unquote
            file_path = unquote(url[7:])

            cwd = os.getcwd()
            abs_path = os.path.abspath(file_path)
            if abs_path.startswith(cwd + os.sep) or abs_path == cwd:
                rel_path = os.path.relpath(abs_path, cwd)
                display_path = rel_path
            else:
                display_path = file_path

        display_args["id"] = display_path
        del display_args["cursor"]

    def _execute_tool_call(self, last_message, tool_instances):
        """Execute a single tool call and return result messages"""
        # Extract tool call information from harmony message
        recipient = str(last_message.recipient)

        # Recipient should be "functions.files_search", "functions.wiki_open", etc.
        # Extract just the function name part for display
        if recipient.startswith("functions."):
            function_name = recipient[10:]  # Remove "functions." prefix for display
        else:
            function_name = recipient

        # Extract arguments from message content
        content_text = last_message.content[0].text if last_message.content else ""

        # Check if tool exists first to determine how to parse content
        tool = tool_instances.get(recipient)

        # Determine if this is a direct-content tool (like PythonTool) or function-based tool
        # Direct-content tools have empty tools array in their config
        is_direct_content_tool = False
        if tool:
            tool_config = tool.tool_config
            if hasattr(tool_config, 'tools') and len(tool_config.tools) == 0:
                is_direct_content_tool = True

        # Parse arguments based on tool type
        if is_direct_content_tool:
            # Direct content tool - content is the raw data (not JSON)
            function_args = {"script": content_text}  # Wrap for display purposes
        else:
            # Function-based tool - content is JSON with arguments
            if not content_text or not content_text.strip():
                function_args = {}
            else:
                parsed = json.loads(content_text)
                # Check if this is a tool call structure with "name" and "arguments"
                # (Granite format) vs just arguments (Harmony format)
                if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                    function_args = parsed["arguments"]
                    # fallback for scaped json
                    if not isinstance(function_args, dict):
                        function_args = json.loads(function_args)
                        
                else:
                    function_args = parsed

        # Format and display tool call with tool class name
        if is_direct_content_tool:
            # For direct content tools, show a summary instead of full content
            content_preview = content_text[:50] + "..." if len(content_text) > 50 else content_text
            args_str = f"script='{content_preview}'"
        else:
            # Build display args, replacing cursor with URL when available
            display_args = dict(function_args)
            self._replace_cursor_with_url_in_display_args(recipient, display_args, tool_instances)
            args_str = ", ".join(f"{k}={repr(v)}" for k, v in display_args.items())

        if recipient in tool_instances:
            tool_class_name = tool_instances[recipient].__class__.__name__
        else:
            tool_class_name = "Unknown"
        blue_print(f"🔧 {tool_class_name}.{function_name}({args_str})")

        # Check if tool exists (use full recipient as key)
        if recipient not in tool_instances:
            # Tool not found - return error to model
            available_tools = ", ".join(sorted(tool_instances.keys()))
            result = {
                "error": "Tool not found",
                "message": f"Tool '{recipient}' is not available. Available tools: {available_tools}"
            }

            result_messages = [Message(
                author=Author(role=Role.TOOL, name=recipient),
                content=[TextContent(text=json.dumps(result))],
            )]
            # Print error in red
            debug_print_error(f"Tool '{recipient}' not found")
            return result_messages

        # Check if tool requires permission and see if user grants it
        requires_permission = tool_class_name in self.restricted_tools

        if (
            requires_permission
            and not self._ask_tool_permission(tool_class_name, recipient, function_args)
        ):
            # Permission denied
            result = {
                "error": "Tool execution cancelled by user",
                "message": f"User declined to execute {recipient}"
            }

            # Return cancelled result message with proper tool metadata
            result_messages = [Message(
                author=Author(role=Role.TOOL, name=recipient),
                content=[TextContent(text=json.dumps(result))],
            )]
            debug_print_error("Cancelled by user")

        else:

            # Execute the tool
            start_function_time = time.time()

            tool = tool_instances[recipient]

            # Create a Message for the tool
            # Strip "functions." prefix so tool receives "files_search" format it expects
            tool_recipient = recipient[10:] if recipient.startswith("functions.") else recipient

            # For direct content tools, pass raw content; for function-based tools, pass JSON args
            if is_direct_content_tool:
                message_content = content_text
            else:
                message_content = json.dumps(function_args)

            tool_message_input = Message(
                author=Author(role=Role.USER, name="user"),
                content=[TextContent(text=message_content)],
            ).with_recipient(tool_recipient)

            # Execute tool asynchronously
            result_messages = []
            try:
                async def run_tool():
                    async for msg in tool._process(tool_message_input):
                        result_messages.append(msg)

                # Run the async tool
                asyncio.run(run_tool())

            except Exception as e:
                # Catch any tool errors and convert to error message
                error_msg = f"{type(e).__name__}: {str(e)}"
                result_messages = [Message(
                    author=Author(role=Role.TOOL, name=recipient),
                    content=[TextContent(text=json.dumps({"error": error_msg}))],
                )]
                debug_print_error(error_msg)
                return result_messages

            elapsed_time = time.time() - start_function_time

            # Count tokens in tool result
            result_text = result_messages[0].content[0].text if result_messages else ""
            result_tokens = self.converter.count_tokens(result_text)

            # Check if result contains an error
            is_error = False
            error_msg = None

            # Try JSON format first
            try:
                result_json = json.loads(result_text)
                if isinstance(result_json, dict) and "error" in result_json:
                    is_error = True
                    error_msg = result_json.get("error", "Unknown error")
            except (json.JSONDecodeError, KeyError):
                # Check for plain text errors (SimpleBrowserTool format)
                if result_text.startswith("Error ") or result_text.startswith("Invalid "):
                    is_error = True
                    error_msg = result_text

            # Print timing and error if present
            blue_print(f"└── {elapsed_time:.1f}s | ~{result_tokens} tokens")
            if is_error:
                debug_print_error(error_msg)

        # Return the actual tool result messages
        return result_messages

    # ── Context compaction ──────────────────────────────────────────

    _COMPACTION_METHODS = {
        "keep_last_n": "_compact_keep_last_n",
        "llm_keep_last_n": "_compact_llm_keep_last_n",
        "llm": "_compact_llm",
    }

    def _compact_keep_last_n(self, messages, original_question):
        """Keep system prefix, the original question, and last N conversation messages."""
        n = self.compaction_keep_n
        prefix = messages[:self._num_system_messages]
        conversation_msgs = messages[self._num_system_messages:]

        question_msg = Message.from_role_and_content(Role.USER, original_question)

        # conversation_msgs[0] is the original user question — skip it since
        # we re-inject it as question_msg to avoid duplication.
        if len(conversation_msgs) <= n + 1:
            return prefix + [question_msg] + conversation_msgs[1:]

        return prefix + [question_msg] + conversation_msgs[-n:]

    def _llm_summarize(self, msgs_to_compact, original_question):
        """Compact a list of messages using the LLM and return a compact Message."""
        context_lines = []
        for msg in msgs_to_compact:
            role = msg.author.role.name
            content = msg.content[0].text if msg.content else ""
            context_lines.append(f"{role}: {content}")

        summary_prompt = (
            "You are summarizing research progress to maintain context within token limits.\n\n"
            f"QUESTION: {original_question}\n\n"
            "Please provide a comprehensive summary of the research context below. "
            "Your summary should:\n"
            "- Preserve ALL specific facts, numbers, names, URLs, and search queries found\n"
            "- Note which tools were called and what results were obtained\n"
            "- Highlight key findings and any dead ends encountered\n"
            "- Keep the summary under 2000 words\n"
            "- Be structured clearly so the research can continue seamlessly\n\n"
            "Context to summarize:\n"
            f"{chr(10).join(context_lines)}"
        )

        summary_messages = [
            Message.from_role_and_content(Role.SYSTEM, "You are a research assistant that summarizes conversation context."),
            Message.from_role_and_content(Role.USER, summary_prompt),
        ]

        result = self.orchestrator.generate_messages(
            messages=summary_messages,
            model=self.model,
            options=self.options,
            timeout=300,
        )

        summary_text = ""
        for msg in result.messages:
            if msg.author.role == Role.ASSISTANT and msg.content:
                summary_text += msg.content[0].text

        return Message.from_role_and_content(
            Role.USER,
            f"[CONTEXT SUMMARY]\n{summary_text}\n\nContinue the research for: {original_question}",
        )

    def _compact_llm_keep_last_n(self, messages, original_question):
        """Compact older messages with LLM, keep last N verbatim."""
        n = self.compaction_keep_n
        prefix = messages[:self._num_system_messages]
        conversation_msgs = messages[self._num_system_messages:]

        if len(conversation_msgs) <= n:
            return prefix + [Message.from_role_and_content(Role.USER, original_question)] + conversation_msgs

        summary_msg = self._llm_summarize(conversation_msgs[:-n], original_question)
        return prefix + [summary_msg] + conversation_msgs[-n:]

    def _compact_llm(self, messages, original_question):
        """Summarize all conversation messages with LLM."""
        prefix = messages[:self._num_system_messages]
        conversation_msgs = messages[self._num_system_messages:]

        if not conversation_msgs:
            return prefix + [Message.from_role_and_content(Role.USER, original_question)]

        summary_msg = self._llm_summarize(conversation_msgs, original_question)
        return prefix + [summary_msg]

    def _compact_context(self, messages, original_question):
        """Dispatch to the configured compaction method."""
        method_name = self._COMPACTION_METHODS.get(self.context_compaction_method)
        if not method_name:
            raise RuntimeError(f"Unknown compaction method: {self.context_compaction_method}")
        return getattr(self, method_name)(messages, original_question)

    # ── Main loop ────────────────────────────────────────────────────

    def user_input(self, prompt, max_iterations=None):
        """Run tool calling loop with retry logic using completions API + harmony"""

        if max_iterations is None:
            max_iterations = self.max_iterations

        context_compaction_count = 0

        # Continue from a copy of conversation history
        # Add prefix if specified (e.g., "Question: ")
        prefixed_prompt = self.input_prefix + prompt
        self.conversation_history.append(Message.from_role_and_content(Role.USER, prefixed_prompt))

        # grow conversation over one or more necessary tool calls
        for iteration in range(max_iterations):

            # Pre-compute prompt for validation and debug
            prompt_string = self.converter.messages_to_prompt_string(self.conversation_history)
            prompt_token_count = self.converter.count_tokens(prompt_string)

            # Context compaction: if enabled and threshold exceeded, try to compact
            if (self.context_compaction_threshold is not None
                    and prompt_token_count > self.context_compaction_threshold
                    and context_compaction_count < self.max_context_compactions):

                blue_print(
                    f"📝 Context ({self._format_tokens(prompt_token_count)}) exceeds threshold "
                    f"({self._format_tokens(self.context_compaction_threshold)}), compacting..."
                )

                self.conversation_history = self._compact_context(
                    self.conversation_history, prefixed_prompt
                )
                context_compaction_count += 1

                # Recompute prompt after compaction
                prompt_string = self.converter.messages_to_prompt_string(self.conversation_history)
                prompt_token_count = self.converter.count_tokens(prompt_string)

                blue_print(f"📝 Context after compaction: {self._format_tokens(prompt_token_count)}")

            # Validate context size
            if prompt_token_count > self.max_context_tokens:
                raise RuntimeError(
                    f"Conversation too long: {self._format_tokens(prompt_token_count)} tokens "
                    f"(max: {self._format_tokens(self.max_context_tokens)}). "
                )

            # Debug print if enabled
            if self.debug:
                self.converter.debug_print_prompt(prompt_string)

            blue_print(f"🤖 {prompt_token_count} tokens input to {self.model}")

            # Generation with retry loop for HarmonyError
            max_retries = 20
            for attempt in range(1, max_retries + 1):
                try:
                    result = self.orchestrator.generate_messages(
                        messages=self.conversation_history,
                        model=self.model,
                        options=self.options,
                        timeout=300
                    )

                    # Success - add messages to history
                    self.conversation_history.extend(result.messages)
                    response_length = result.response_token_count
                    elapsed_time = result.elapsed_time
                    break

                except Backend500Error as e:
                    # Backend server error - format with context
                    error_msg = (
                        f"Backend server error (500). This may be due to:\n"
                        f"  • Large context size (current: {self._format_tokens(e.prompt_token_count)} tokens)\n"
                        f"  • Model out of memory\n"
                        f"  • Backend server issue\n"
                        f"Try reducing tool output size or restarting the backend."
                    )
                    print(f"\n\033[91m✗ Error: {error_msg}\033[0m\n")
                    self._save_conversation_log()
                    self._save_out_messages()
                    raise Backend500Error(error_msg, e.prompt_token_count)

                except BackendHTTPError as e:
                    # Other HTTP errors
                    error_msg = f"Backend returned status {e.status_code}: {e.response_text}"
                    print(f"\n\033[91m✗ Error: {error_msg}\033[0m\n")
                    self._save_conversation_log()
                    self._save_out_messages()
                    return "I encountered a backend error and cannot continue. Please check the backend status."

                except BackendConnectionError as e:
                    # Connection failures
                    error_msg = str(e)
                    print(f"\n\033[91m✗ Error: {error_msg}\033[0m\n")
                    self._save_conversation_log()
                    self._save_out_messages()
                    return "I cannot connect to the backend. Please check if the backend is running."

                except GenerationError as e:
                    # Unexpected generation errors
                    error_msg = str(e)
                    print(f"\n\033[91m✗ Error: {error_msg}\033[0m\n")
                    self._save_conversation_log()
                    self._save_out_messages()
                    return "I encountered an unexpected error and cannot continue."

                except HarmonyError as he:
                    # Parsing error - retry if attempts remain
                    if attempt < max_retries:
                        print(f"\n\033[93m⚠ Parse error on attempt {attempt}/{max_retries}, retrying generation...\033[0m")
                        print(f"   Error: {str(he)[:100]}")
                        continue
                    # Max retries exceeded
                    print(f"\n\033[91m✗ Failed to generate valid format after {max_retries} attempts\033[0m")
                    print(f"   Last error: {str(he)}")
                    self._save_conversation_log()
                    self._save_out_messages()
                    raise

            blue_print(f"└── {elapsed_time:.1f}s | ~{response_length} tokens generated")

            last_message = self.conversation_history[-1]

            # Check if model wants to call a tool (check for recipient field in harmony messages)
            if hasattr(last_message, 'recipient') and last_message.recipient:
                # Execute the tool call and get result messages
                tool_result_messages = self._execute_tool_call(last_message, self.tool_instances)
                self.conversation_history.extend(tool_result_messages)
                # Continue loop to process tool results

            # Check if assistant marked response as final (proper harmony stop condition)
            elif (last_message.author.role == Role.ASSISTANT and
                  getattr(last_message, "channel", None) == "final"):
                # Extract text content from last message
                response_text = last_message.content[0].text
                self._save_conversation_log()
                self._save_out_messages()
                return response_text

            # If neither tool call nor final, assume it's final (fallback for compatibility)
            elif last_message.author.role == Role.ASSISTANT:
                # Extract text content from last message
                response_text = last_message.content[0].text
                self._save_conversation_log()
                self._save_out_messages()
                return response_text

        # If we reach here, model didn't finish in max_iterations
        # Update conversation history and ask user if they need help
        self._save_conversation_log()
        self._save_out_messages()
        return self.LOOP_TOOL_FAIL
