"""Tests for the structured-decoding stop condition (rl.policy.first_json_object_end).

Pure Python — no torch, no model, no GPU. Importing rl.policy must NOT pull in
unsloth/transformers (those imports are function-local in LLMPolicy.__init__), so
this also guards that the module stays cheap to import on a CPU box.

Run anywhere:  python tests/test_json_stop.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.policy import first_json_object_end  # noqa: E402


def _slice(text):
    """Return the substring up to the detected end of the first JSON object."""
    end = first_json_object_end(text)
    return None if end is None else text[:end]


def test_simple_object():
    s = '{"a": 1}'
    assert first_json_object_end(s) == len(s)


def test_incomplete_object_returns_none():
    assert first_json_object_end('{"a": 1') is None
    assert first_json_object_end('{"clients": [{"id": 0') is None
    assert first_json_object_end("") is None
    assert first_json_object_end("no json here") is None


def test_nested_object_needs_outer_close():
    s = '{"clients": [{"id": 0, "operations": [{"op": "scale"}]}]}'
    # Must return the OUTER close, not the first inner "}".
    assert first_json_object_end(s) == len(s)
    # Truncated right after an inner close is still incomplete.
    trunc = '{"clients": [{"id": 0}'
    assert first_json_object_end(trunc) is None


def test_trailing_text_after_object_is_cut():
    s = '{"a": 1} then the model kept talking...'
    end = first_json_object_end(s)
    assert s[:end] == '{"a": 1}'


def test_braces_inside_strings_are_ignored():
    # A "}" inside a string value must not close the object early.
    s = '{"reason": "looks like a }{ mess", "is_suspicious": true}'
    assert _slice(s) == s
    # Opening brace inside a string must not raise depth either.
    s2 = '{"reason": "a { b", "ok": false}'
    assert _slice(s2) == s2


def test_escaped_quote_inside_string():
    # Escaped quote must not terminate the string early (so the following } counts
    # as inside the string until the real closing quote).
    s = '{"reason": "he said \\"hi}\\" ok", "v": 1}'
    assert _slice(s) == s


def test_markdown_fence_preamble_is_skipped():
    s = '```json\n{"clients": []}\n```'
    end = first_json_object_end(s)
    assert s[:end] == '```json\n{"clients": []}'  # stops right after the object close


def test_defender_style_verdict_list():
    s = ('{"clients": ['
         '{"client_id": 0, "is_suspicious": true, "confidence": 0.9, "reason": "big"},'
         '{"client_id": 1, "is_suspicious": false, "confidence": 0.1, "reason": "ok"}'
         ']}')
    assert first_json_object_end(s) == len(s)
    # One char short of the outer close → not done yet.
    assert first_json_object_end(s[:-1]) is None


def test_attacker_style_plan():
    s = '{"clients": [{"id": 2, "operations": [{"op": "sign_flip", "target": "all"}]}]}'
    assert first_json_object_end(s) == len(s)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} json-stop tests passed.")


if __name__ == "__main__":
    _run()
