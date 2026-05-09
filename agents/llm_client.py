"""Thin OpenAI chat completion wrapper with JSON response parsing."""

import json
import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """Minimal wrapper around OpenAI chat completions."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def call(self, system_prompt: str, user_message: str) -> dict:
        """Send a chat completion request and return parsed JSON."""
        logger.info(f"LLM call → model={self.model}")
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
            logger.error(f"LLM call failed: {e}")
            return {}
