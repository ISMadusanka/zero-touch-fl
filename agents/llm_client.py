"""LLM client abstraction — supports OpenAI and Ollama backends.

Backend selection is driven by the ``llm_backend`` key injected into each
agent's config at runtime (see ``main.py --env``).

* ``openai``  — uses the OpenAI Python SDK (chat completions, JSON mode).
* ``ollama``  — sends HTTP requests to a local Ollama server
               (``POST /api/generate``).
"""

import abc
import json
import logging
import os
import re

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLMClient(abc.ABC):
    """Interface every LLM backend must implement."""

    @abc.abstractmethod
    def call(self, system_prompt: str, user_message: str) -> dict:
        """Send a prompt and return the parsed JSON response."""


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAILLMClient(BaseLLMClient):
    """OpenAI chat-completion client with JSON-mode response parsing."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def call(self, system_prompt: str, user_message: str) -> dict:
        logger.info(f"LLM call [OpenAI] → model={self.model}")
        logger.debug(f"System: {system_prompt[:200]}...")
        logger.debug(f"User: {user_message[:200]}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            content = response.choices[0].message.content
            logger.debug(f"LLM response: {content[:300]}")
            return json.loads(content)
        except Exception as e:
            logger.error(f"OpenAI LLM call failed: {e}")
            return {}


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaLLMClient(BaseLLMClient):
    """Ollama local-model client using the ``/api/generate`` endpoint."""

    def __init__(
        self,
        model: str = "deepseek-r1:70b",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")

    def call(self, system_prompt: str, user_message: str) -> dict:
        logger.info(f"LLM call [Ollama] → model={self.model} @ {self.base_url}")
        logger.debug(f"System: {system_prompt[:200]}...")
        logger.debug(f"User: {user_message[:200]}...")

        # Combine system + user into a single prompt for the generate API
        prompt = (
            f"[System Instructions]\n{system_prompt}\n\n"
            f"[User Message]\n{user_message}\n\n"
            "Respond with ONLY a valid JSON object. No extra text."
        )

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=300,
            )
            resp.raise_for_status()

            raw_response = resp.json().get("response", "")
            logger.debug(f"LLM response: {raw_response[:300]}")
            return self._extract_json(raw_response)
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is the Ollama server running?"
            )
            return {}
        except Exception as e:
            logger.error(f"Ollama LLM call failed: {e}")
            return {}

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract the first JSON object from model output.

        Ollama models may wrap the JSON in markdown fences or include
        chain-of-thought text before the actual object.

        Uses brace-counting to correctly handle nested JSON structures
        (e.g. weight dicts containing nested arrays/objects).
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code fences (greedy to capture nested braces)
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # Brace-counting: find the outermost { ... } block
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
                        break

        logger.warning("Could not extract valid JSON from Ollama response")
        return {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client(
    backend: str = "openai",
    model: str | None = None,
    temperature: float = 0.7,
    ollama_base_url: str = "http://localhost:11434",
) -> BaseLLMClient:
    """Create the appropriate LLM client based on the chosen backend.

    Parameters
    ----------
    backend : str
        ``"openai"`` or ``"ollama"``.
    model : str | None
        Model identifier.  Defaults to ``"gpt-4o-mini"`` for OpenAI and
        ``"deepseek-r1:70b"`` for Ollama when *None*.
    temperature : float
        Sampling temperature.
    ollama_base_url : str
        Base URL of the Ollama server (only used when *backend* is
        ``"ollama"``).
    """
    backend = backend.lower().strip()

    if backend == "ollama":
        resolved_model = model or "deepseek-r1:70b"
        logger.info(f"Using Ollama backend → {resolved_model} @ {ollama_base_url}")
        return OllamaLLMClient(
            model=resolved_model,
            temperature=temperature,
            base_url=ollama_base_url,
        )

    # Default: OpenAI
    resolved_model = model or "gpt-4o-mini"
    logger.info(f"Using OpenAI backend → {resolved_model}")
    return OpenAILLMClient(model=resolved_model, temperature=temperature)
