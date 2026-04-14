"""Main entry point for droplet"""

import inspect
import os
import subprocess
import sys

import droplet.tools as tools_module
from droplet import dbg_tools
from droplet.agent import DropletAgent
from droplet.config_manager import list_configs, load_config, save_config
from droplet.rich_cl import get_user_input
from droplet.rich_help import create_argument_parser_with_rich_help
from droplet.rich_terminal import LOGO_FAILURE, droplet_print, print_logo
from droplet.rits_utils import list_rits_models_and_exit, resolve_api_key


def build_agent_config():
    """
    Parse command line arguments, load config, and return a dict ready for DropletAgent(**config).

    Returns:
        tuple: (agent_config_dict, backend_name_for_display, initial_input, cwd) or (None, None, None, None) if should exit
    """
    # Discover all *Tool classes from droplet.tools
    excluded_tools = {'SimpleFunctionTool', 'BrowseTool'}
    available_tools = {}
    for name, obj in inspect.getmembers(tools_module, inspect.isclass):
        if name.endswith('Tool') and name not in excluded_tools:
            if hasattr(obj, '__module__') and obj.__module__.startswith('droplet'):
                available_tools[name] = obj

    # Create parser with rich help formatting
    parser = create_argument_parser_with_rich_help(available_tools)

    # Backend configuration
    parser.add_argument('-b', '--backend-type', type=str, default='ollama',
                       choices=['ollama', 'vllm', 'rits-vllm', 'llama-cpp'],
                       help='Backend type (default: ollama)')
    parser.add_argument('-u', '--backend-url', type=str, default='http://localhost:11434',
                       help='Backend URL (default: http://localhost:11434 for Ollama, ignored for rits-vllm)')
    parser.add_argument('-m', '--model', type=str, default='gpt-oss:20b',
                       help='Model name (default: gpt-oss:20b)')
    parser.add_argument('--rits-api-key', type=str,
                       help='RITS API key (required for rits-vllm backend)')
    parser.add_argument('--rits-list-models', action='store_true',
                       help='List all available RITS models and exit')
    parser.add_argument('--semantic-scholar-api-key', type=str,
                       help='Semantic Scholar API key (optional, provides higher rate limits)')
    parser.add_argument('-t', '--tools', type=str, nargs='+',
                       default=['FileBrowserTool', 'SemanticScholarTool', 'PythonTool', 'WikipediaBrowserTool'],
                       help=f'Tools to use (default: FileBrowserTool, WikipediaBrowserTool, SemanticScholarTool, PythonTool). '
                            f'Available: {", ".join(sorted(available_tools.keys()))}')

    # Generation parameters
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (default: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=32768,
                       help='Maximum tokens to generate (default: 32768)')
    parser.add_argument('--max-iterations', type=int, default=25,
                       help='Maximum tool call iterations per user input (default: 10)')

    # Droplet-specific arguments
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--log', type=str, help='Path to JSON file to log conversation history')
    parser.add_argument('--out-messages', type=str, help='Path to JSON file to write final message list (for batch evaluation)')
    parser.add_argument('-i', '--input', type=str, dest='initial_input',
                       help='Initial prompt to send to the agent')
    parser.add_argument('--require-approval', type=str, nargs='*',
                       default=['WikipediaBrowserTool', 'PythonTool'],
                       help='Tools that require user approval before execution (default: WikipediaBrowserTool, PythonTool)')
    parser.add_argument('--cwd', type=str,
                       help='Change working directory before starting agent (agent will operate from this directory)')

    # Milvus specific flags
    parser.add_argument('--milvus-db', type=str, help='Path to Milvus database file (required for RetrieverBrowserTool)')
    parser.add_argument('--milvus-model', type=str,
                       default=None,
                       help='SentenceTransformer model name or path (default: sentence-transformers/all-MiniLM-L6-v2)')
    parser.add_argument('--milvus-collection', type=str,
                       default=None,
                       help='Milvus collection name (default: nq_train_short_granite149m)')

    # llama.cpp specific flags
    parser.add_argument('--n-gpu-layers', type=int, default=None,
                       help='Number of layers to offload to GPU for llama-cpp backend (default: auto-detect)')
    parser.add_argument('--n-ctx', type=int, default=8192,
                       help='Context window size for llama-cpp backend (default: 8192)')
    parser.add_argument('--gguf-file', type=str, default=None,
                       help='Specific GGUF filename to download from HuggingFace repo')

    # BCP specific flags
    parser.add_argument('--bcp-server-url', type=str, help='BCP search server URL (required for BCPBrowserTool, default: http://localhost:8000)')

    # Prompt configuration
    parser.add_argument('--no-droplet-system-prompt', action='store_true',
                       help='Disable default Droplet system prompt (use with --system-prompt to provide custom prompt)')
    parser.add_argument('--system-prompt', type=str,
                       help='Custom system prompt (only used when --no-droplet-system-prompt is set)')
    parser.add_argument('--developer-prompt', type=str,
                       help='Additional developer instructions added as developer message')
    parser.add_argument('--loop-tool-fail', type=str,
                       help='Override default loop failure message')
    parser.add_argument('--input-prefix', type=str,
                       help='Prefix to add to user input messages (e.g., "Question: ")')
    parser.add_argument('--gpt-reasoning', type=str, choices=['low', 'medium', 'high'],
                       help='GPT-OSS reasoning effort level: low, medium, or high (default: model default)')

    # Context compaction
    parser.add_argument('--context-compaction-method',
                       choices=['keep_last_n', 'llm_keep_last_n', 'llm'],
                       default=None,
                       help='Context compaction method (default: None = disabled)')
    parser.add_argument('--context-compaction-threshold', type=int, default=64000,
                       help='Token threshold to trigger context compaction (default: 64000)')
    parser.add_argument('--max-context-compactions', type=int, default=3,
                       help='Max compaction rounds per user input (default: 3)')
    parser.add_argument('--compaction-keep-n', type=int, default=5,
                       help='Number of recent messages to keep after compaction, must be >= 1 (default: 5)')

    # Configuration management
    parser.add_argument('-c', '--load-config', type=str, metavar='NAME',
                       help='Load a saved configuration')
    parser.add_argument('-l', '--list-configs', action='store_true',
                       help='List all saved configurations and exit')
    parser.add_argument('-s', '--save-config', type=str, nargs='?', const='__DEFAULT__', default='__NOT_SET__', metavar='NAME',
                       help='Save current configuration (use without name for default config)')

    # Parse arguments
    args = parser.parse_args()

    # Handle --list-configs first
    if args.list_configs:
        list_configs()
        sys.exit(0)

    # Get default values from parser for comparison
    defaults = {}
    for action in parser._actions:
        if action.dest != 'help':
            defaults[action.dest] = action.default

    # Determine which arguments were explicitly provided on command line
    explicitly_provided = set()
    for action in parser._actions:
        if action.dest == 'help':
            continue
        for opt in action.option_strings:
            if opt in sys.argv:
                explicitly_provided.add(action.dest)
                break

    # Store original args before config loading
    original_args = {}
    for key, value in vars(args).items():
        original_args[key] = value

    # Determine which config to load
    config_name = args.load_config if args.load_config else "None"
    config = load_config(config_name)

    if config:
        # Apply config values only for arguments NOT explicitly provided on command line
        for key, value in config.items():
            if key not in explicitly_provided:
                setattr(args, key, value)
    elif args.load_config:
        # Only error if user explicitly requested a config that doesn't exist
        print(f"\n\033[91mError: Configuration '{args.load_config}' not found\033[0m")
        print("Use --list-configs to see available configurations\n")
        sys.exit(1)

    # Store original args for potential saving
    args._original_args = original_args
    args._defaults = defaults
    args._explicitly_provided = explicitly_provided

    # Handle --save-config flag
    if args.save_config != '__NOT_SET__':
        config_name = None if args.save_config == '__DEFAULT__' else args.save_config
        save_config(config_name, args)
        # Continue with normal execution after saving

    # Handle --rits-list-models flag
    if args.rits_list_models:
        list_rits_models_and_exit(args)
        return None, None, None, None

    # Determine backend name for logo display
    if args.backend_type == "rits-vllm":
        backend_name = "RITSBackend (https://rits.fmaas.res.ibm.com)"
    elif args.backend_type == "vllm":
        backend_name = f"VLLMBackend ({args.backend_url})"
    elif args.backend_type == "ollama":
        backend_name = f"OllamaBackend ({args.backend_url})"
    elif args.backend_type == "llama-cpp":
        backend_name = f"LlamaCppBackend (Metal)"
    else:
        raise Exception(f"Unknown backend {args.backend_type}")

    # Resolve API key for RITS endpoints
    # For rits-vllm backend, check if API key needs resolution
    if args.backend_type == "rits-vllm":
        base_url_for_check = "https://rits.fmaas.res.ibm.com"
    else:
        base_url_for_check = args.backend_url

    args.rits_api_key = resolve_api_key(args.rits_api_key, base_url_for_check)

    # Build clean config dict for DropletAgent
    agent_config = {
        'model': args.model,
        'base_url': args.backend_url,
        'backend_type': args.backend_type,
        'debug': args.debug,
        'restricted_tools': set(args.require_approval),
        'tool_names': args.tools,
        'milvus_db': args.milvus_db,
        'milvus_model': args.milvus_model,
        'milvus_collection': args.milvus_collection,
        'bcp_server_url': args.bcp_server_url,
        'log_file': args.log,
        'out_messages': args.out_messages,
        'rits_api_key': args.rits_api_key,
        'semantic_scholar_api_key': args.semantic_scholar_api_key,
        'no_droplet_sytem_prompt': args.no_droplet_system_prompt,
        'system_prompt': args.system_prompt,
        'developer_prompt': args.developer_prompt,
        'loop_tool_fail': args.loop_tool_fail,
        'input_prefix': args.input_prefix,
        'gpt_reasoning': args.gpt_reasoning,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'max_iterations': args.max_iterations,
        'context_compaction_method': args.context_compaction_method,
        'context_compaction_threshold': args.context_compaction_threshold,
        'max_context_compactions': args.max_context_compactions,
        'compaction_keep_n': args.compaction_keep_n,
        'n_gpu_layers': args.n_gpu_layers,
        'n_ctx': args.n_ctx,
        'gguf_file': args.gguf_file,
    }

    # Return cwd separately (not part of agent config)
    return agent_config, backend_name, args.initial_input, args.cwd


def main():
    """Droplet entrypoint - initializes backend (Ollama or vLLM) and agent"""
    # Build consolidated agent configuration
    agent_config, backend_name, initial_input, cwd = build_agent_config()

    # If None returned, special flag was handled (--list-configs, --rits-list-models)
    if agent_config is None:
        return

    # Resolve file paths to absolute paths BEFORE changing directories
    # This ensures relative paths are resolved from the original cwd
    if agent_config.get('log_file'):
        agent_config['log_file'] = os.path.abspath(os.path.expanduser(agent_config['log_file']))

    if agent_config.get('milvus_db'):
        agent_config['milvus_db'] = os.path.abspath(os.path.expanduser(agent_config['milvus_db']))

    # Change working directory if --cwd was specified
    if cwd:
        cwd = os.path.expanduser(cwd)  # Expand ~ to home directory
        if not os.path.isdir(cwd):
            print(f"\n\033[91mError: Directory '{cwd}' does not exist\033[0m\n")
            sys.exit(1)
        os.chdir(cwd)
        print(f"\033[94m📁 Working directory: {os.getcwd()}\033[0m\n")

    if agent_config['debug']:
        dbg_tools.pm_breakpoint()

    try:
        # Clear screen and move cursor to top
        print("\033[2J\033[H", end="", flush=True)

        # Print logo first, then initialize agent
        print_logo(model=agent_config['model'], backend=backend_name, tools=agent_config['tool_names'])
        print()

        # Print helpful message in grey
        print("\033[90mYou can ask DROP to summarize current folder or look for specific information using the tools above. You can use '!' to run a conventional command line that DROP does not see e.g. !ls\033[0m\n")

        # initial_input already extracted from build_agent_config() return value
        # (it's not in agent_config dict)

        # Initialize agent with consolidated config
        try:
            agent = DropletAgent(**agent_config)
        except RuntimeError as e:
            error_msg = str(e)

            # Clean up HTML error pages
            if "<html" in error_msg.lower():
                error_msg = "Service unavailable (received HTML error page)"

            # Clear screen and show failure logo
            print("\033[2J\033[H", end="", flush=True)

            # For RITS errors, show logo with both model and backend in red
            if agent_config['backend_type'] == "rits-vllm":
                print_logo(
                    model=f"\033[91m{agent_config['model']}\033[0m",
                    backend=f"\033[91m{backend_name}\033[0m",
                    tools=agent_config['tool_names'],
                    logo=LOGO_FAILURE
                )
                # Show specific error message if it's about missing API key
                if "RITS API key is required" in error_msg or "rits-api-key" in error_msg.lower():
                    print("\n\033[91mError: Missing --rits-api-key\033[0m\n")
                else:
                    print(f"\n\033[91mError: {error_msg}\033[0m\n")
            # For other errors, show logo with backend in red
            else:
                print_logo(
                    model=f"\033[91m{agent_config['model']}\033[0m",
                    backend=f"\033[91m{backend_name}\033[0m",
                    tools=agent_config['tool_names'],
                    logo=LOGO_FAILURE
                )
                print(f"\n\033[91mError: {error_msg}\033[0m\n")
            return

        with agent:
            # Handle initial prompt if provided
            if initial_input:
                response = agent.user_input(initial_input)
                droplet_print(response)

            # Simple interactive loop
            while True:
                user_input, is_system_command = get_user_input()

                if user_input.lower() in ["exit", "quit"]:
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue

                # Handle system commands
                if is_system_command:
                    try:
                        subprocess.run(
                            user_input,
                            shell=True,
                            capture_output=False,
                            text=True
                        )
                    except Exception as e:
                        print(f"\033[91mCommand failed: {e}\033[0m")
                    continue

                else:

                    # Handle agent queries
                    response = agent.user_input(user_input)
                    droplet_print(response)

    except (KeyboardInterrupt, EOFError):
        print("\n\nGoodbye!")
        return


if __name__ == "__main__":
    main()
