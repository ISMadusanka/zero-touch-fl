"""Best-effort JSON extraction from raw LLM output.

Small instruct models wrap their JSON in prose, markdown fences, or both, even
when the system prompt forbids it. Failing to parse such a response would throw
away an otherwise perfectly good verdict, so the parse is layered: strict, then
fenced, then the first brace-to-brace span.

Previously lived in ``agents/attack_ops.py`` alongside the weight-space attack
DSL. That module is gone (the attack is now label flipping — see
:mod:`data.label_flip`), but the defender's output parser still needs this.
"""

import json
import re


def extract_json(text):
    """Return parsed JSON from ``text``, or ``None`` if nothing usable is there.

    Already-parsed input (a dict or list) is passed straight through, so callers
    can hand this either a raw completion or a structured stub.
    """
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    fence = re.search(r"```(?:json)?\s*([\[{].*[\]}])\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    brace = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    return None
