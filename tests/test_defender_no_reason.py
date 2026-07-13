"""Tests that the defender output schema dropped the `reason` field (to cut
generated tokens) while the parser stays backward-compatible.

Run on any box:  python tests/test_defender_no_reason.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.defender_agent import DefenderAgent, SYSTEM_PROMPT  # noqa: E402


def test_prompt_schema_has_no_reason_field():
    # The JSON schema the model is shown must be the three-field form.
    assert '"is_suspicious": <true|false>, "confidence": <float 0..1>}' in SYSTEM_PROMPT
    # And it must explicitly forbid a reason field.
    assert 'do NOT add a "reason"' in SYSTEM_PROMPT
    # The old 4-field schema line must be gone.
    assert '"confidence": <float 0..1>, "reason"' not in SYSTEM_PROMPT


def test_parse_reasonless_output():
    agent = DefenderAgent()
    text = json.dumps({"clients": [
        {"client_id": 0, "is_suspicious": True, "confidence": 0.9},
        {"client_id": 1, "is_suspicious": False, "confidence": 0.2},
    ]})
    verdicts = agent.parse(text, [0, 1])
    assert [v.client_id for v in verdicts] == [0, 1]
    assert verdicts[0].is_suspicious is True and abs(verdicts[0].confidence - 0.9) < 1e-9
    assert verdicts[1].is_suspicious is False
    # No reason emitted → defaults to "" (never None), so logging stays safe.
    assert verdicts[0].reason == ""


def test_parse_still_tolerates_reason_if_present():
    # Backward compatibility: a stray reason field must not break parsing.
    agent = DefenderAgent()
    text = json.dumps({"clients": [
        {"client_id": 0, "is_suspicious": True, "confidence": 0.8, "reason": "big norm"},
    ]})
    verdicts = agent.parse(text, [0])
    assert verdicts[0].is_suspicious is True
    assert verdicts[0].reason == "big norm"


def test_missing_client_defaults_benign():
    agent = DefenderAgent()
    text = json.dumps({"clients": [{"client_id": 0, "is_suspicious": True, "confidence": 0.9}]})
    verdicts = agent.parse(text, [0, 1, 2])       # 1 and 2 omitted by the model
    assert len(verdicts) == 3
    assert verdicts[1].is_suspicious is False and verdicts[1].reason == "unparsed"


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} defender-no-reason tests passed.")


if __name__ == "__main__":
    _run()
