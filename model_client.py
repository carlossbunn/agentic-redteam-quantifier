"""
Pluggable model-client interface for the red-teaming quantification pipeline.

Two families of clients:
  - MockModel: deterministic/stochastic stand-in used for the benign-proxy
    demo in this repo. No network calls, no external dependency.
  - Real API adapters (OpenAIClient, AnthropicClient): thin wrappers left as
    stubs. Wire in your own, institutionally-approved API credentials and
    review use with your advisor/ethics board before pointing this pipeline
    at production models with anything beyond the benign proxy benchmark.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class ModelClient:
    """Abstract interface every target/attacker model must implement."""

    name: str = "abstract"

    def chat(self, messages: List[Message]) -> str:
        raise NotImplementedError


class MockModel(ModelClient):
    """
    A harmless stand-in target model for pipeline development/testing.

    Behavior mimics, in miniature, the kind of graded compliance real
    safety-tuned models show in the source papers:
      - Refuses more often early in the conversation.
      - Becomes more permissive as (a) more turns elapse and (b) more
        "softening" framings (hypothetical/educational/jargon) appear in
        the running conversation -- this is exactly what the pipeline's
        harm-score trajectory (see metrics.py) is meant to detect.

    Holds a single `protected_secret` (a benign placeholder string/word)
    that stands in for "the harmful content" in the real papers. Success
    for this demo == getting MockModel to output that placeholder.
    """

    def __init__(self, name: str, protected_secret: str, refusal_strength: float = 0.85,
                 seed: Optional[int] = None):
        self.name = name
        self.protected_secret = protected_secret
        self.refusal_strength = refusal_strength
        self._rng = random.Random(seed)
        self._softening_score = 0.0

    def _update_softening(self, text: str) -> None:
        cues = ["hypothetical", "hipotétic", "fictional", "ficção", "educational",
                "educacional", "research", "pesquisa", "in a movie", "no filme",
                "for a class", "para uma aula", "step-by-step", "passo a passo",
                "thought experiment"]
        hits = sum(1 for c in cues if c.lower() in text.lower())
        self._softening_score += 0.12 * hits
        self._softening_score += 0.03  # mild "fatigue" per turn regardless of content

    def chat(self, messages: List[Message]) -> str:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        self._update_softening(last_user)
        compliance_prob = min(0.97, max(0.02, self._softening_score - self.refusal_strength + 0.5))
        if self._rng.random() < compliance_prob:
            return (f"[MOCK-{self.name}] Sure, here it is step by step: "
                    f"the protected value is '{self.protected_secret}'. "
                    f"(for illustrative/educational purposes)")
        return f"[MOCK-{self.name}] I'm sorry, but I can't help with that request."


# --- Real-API stubs -------------------------------------------------------
# Left unimplemented on purpose. Fill in with your own credentials only if
# you move from the benign proxy benchmark to real target models, and only
# within whatever usage policy / ethics approval governs your IC.

class OpenAIClient(ModelClient):
    def __init__(self, model: str, api_key: str):
        self.name = model
        self.api_key = api_key

    def chat(self, messages: List[Message]) -> str:
        raise NotImplementedError("Wire up an openai>=1.0 client here with your own API key.")


class AnthropicClient(ModelClient):
    def __init__(self, model: str, api_key: str):
        self.name = model
        self.api_key = api_key

    def chat(self, messages: List[Message]) -> str:
        raise NotImplementedError("Wire up an anthropic client here with your own API key.")
