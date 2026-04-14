"""Registry and factory for message converters"""

from droplet.converters.granite import GraniteMessageConverter
from droplet.converters.harmony import HarmonyMessageConverter

# Pattern-based model to converter mapping
CONVERTER_PATTERNS = {
    "gpt-oss": HarmonyMessageConverter,
    "granite": GraniteMessageConverter,
    "ibm-granite": GraniteMessageConverter,
}


def get_converter_for_model(model_name):
    """
    Get appropriate message converter for model

    Args:
        model_name: Name of the model

    Returns:
        MessageConverter instance

    Raises:
        ValueError: If no converter found for model
    """
    # Strip -GGUF suffix for converter lookup (GGUF repos don't have tokenizer files)
    converter_model_name = model_name
    if converter_model_name.upper().endswith("-GGUF"):
        converter_model_name = converter_model_name[:-5]

    # Match pattern in model name
    for pattern, converter_class in CONVERTER_PATTERNS.items():
        if pattern in model_name.lower():
            # Instantiate converter
            if converter_class == HarmonyMessageConverter:
                # HarmonyMessageConverter takes no args
                return converter_class()
            elif converter_class == GraniteMessageConverter:
                # GraniteMessageConverter needs model name for tokenizer
                return converter_class(converter_model_name)

    # No match found
    supported_patterns = ", ".join(CONVERTER_PATTERNS.keys())
    raise ValueError(
        f"No converter found for model '{model_name}'. "
        f"Model name must contain one of: {supported_patterns}"
    )
