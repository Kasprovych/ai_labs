"""Large language model interface and the local MLX implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT = "system", "user", "assistant"


@dataclass(frozen=True)
class ChatMessage:
    """One message of a chat prompt."""
    role: str
    content: str


class ILanguageModel(ABC):
    @abstractmethod
    def generate(self, messages: Sequence[ChatMessage], max_tokens: int) -> str:
        """Returns the assistant reply for the given chat history."""


class MlxLanguageModel(ILanguageModel):
    """A quantized open-weights LLM running locally on Apple Silicon through the MLX framework.

    The model is downloaded from Hugging Face Hub on first use and cached in ~/.cache/huggingface.
    The chat template stored with the model converts the message list into the exact prompt
    format (special tokens for roles) the model was trained on.
    """

    def __init__(self, model_name: str, temperature: float) -> None:
        from mlx_lm import load  # heavy import, kept local
        from mlx_lm.sample_utils import make_sampler

        self._model, self._tokenizer = load(model_name)
        self._sampler = make_sampler(temp=temperature)
        self.model_name = model_name

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def generate(self, messages: Sequence[ChatMessage], max_tokens: int) -> str:
        from mlx_lm import generate

        prompt = self._tokenizer.apply_chat_template(
            [{"role": message.role, "content": message.content} for message in messages],
            add_generation_prompt=True, tokenize=False)
        return generate(self._model, self._tokenizer, prompt=prompt, max_tokens=max_tokens,
                        sampler=self._sampler, verbose=False).strip()


class ScriptedLanguageModel(ILanguageModel):
    """Test double: returns pre-defined replies and records the prompts it received."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.calls: List[Sequence[ChatMessage]] = []

    def generate(self, messages: Sequence[ChatMessage], max_tokens: int) -> str:
        self.calls.append(messages)
        return self._replies[min(len(self.calls), len(self._replies)) - 1]
