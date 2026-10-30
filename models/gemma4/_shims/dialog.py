"""Stand-in for the `dialog` chat-format package (tokenizer class attribute only)."""
import enum


class Format(enum.Enum):
    GEMMA2 = "gemma2"
    GEMMA3 = "gemma3"
    GEMMA4 = "gemma4"
