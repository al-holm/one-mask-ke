from enum import Enum


class ModelName(Enum):
    """
    Enum for model names.
    """

    GPT2XL = "gpt2-xl"
    LLAMA3_3B = "llama3-3b"
    MISTRAL7B = "mistral-7b"

    def __str__(self):
        return self.value
