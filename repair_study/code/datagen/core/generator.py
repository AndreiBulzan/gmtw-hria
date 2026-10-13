"""Generator model client: any OpenAI-compatible /v1/chat/completions
endpoint (vLLM serving Llama-3.3-70B / Qwen2.5-72B, etc.). stdlib-only.

Bulk generation stays on open-weight models (ToS); frontier APIs are for
small audit samples only — enforced by policy, not by this client."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


DEFAULT_MODEL = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"


@dataclass
class GeneratorConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: str = "EMPTY"
    timeout: float = 300.0

    @property
    def provenance(self) -> str:
        return self.model


class Generator:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg

    def chat(self, prompt: str, *, temperature: float | None = None) -> str:
        return self.chat_n(prompt, 1, temperature=temperature)[0]

    def chat_n(self, prompt: str, n: int, *,
               temperature: float | None = None) -> list[str]:
        """n completions for one prompt, in a single request.

        Asking the server for n rather than looping costs one prefill instead
        of n, which matters here: the constraint prompts carry a catalogue and
        a schema, so the prompt dominates the work. At temperature 0 the n
        completions come back identical -- that is the intended control, not a
        bug."""
        body = json.dumps({
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens,
            "n": n,
        }).encode()
        req = urllib.request.Request(
            f"{self.cfg.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.cfg.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
            data = json.load(resp)
        return [c["message"]["content"] for c in data["choices"]]
