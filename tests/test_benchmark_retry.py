"""Tests for the benchmark harness's unusable-attacker-action handling.

A round whose attacker generation does not fill the exact poison quota — truncated
JSON, an empty plan, an all-no-op plan — used to abort the whole benchmark with a
RuntimeError, discarding every round measured so far. It is now resampled a bounded
number of times and, only if still unusable, skipped and excluded from the metrics.

Covers benchmark.harness._sample_attack + the skip path in run_benchmark. Needs
torch (the harness imports the FL server).

Run:  python tests/test_benchmark_retry.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.attacker_agent import AttackerAgent          # noqa: E402
from benchmark import harness                            # noqa: E402
from benchmark.harness import _hit_token_cap, _sample_attack, _snippet, run_benchmark  # noqa: E402
from core.types import DetectionVerdict, ModelUpdate     # noqa: E402

# Unusable-action warnings are the EXPECTED path through most of these tests, so keep
# them out of the test output; the one test that asserts on them attaches its own
# handler to this logger.
logging.getLogger("benchmark").addHandler(logging.NullHandler())
logging.getLogger("benchmark").propagate = False

_GOOD = ('{"clients":[{"id":0,"operations":[{"op":"scale","target":"all","factor":2.0}]},'
         '{"id":1,"operations":[{"op":"scale","target":"all","factor":0.5}]}]}')
_TRUNCATED = '{"clients":[{"id":0,"operations":[{"op":"scale","target":"all","fac'
_NOOP = '{"clients":[{"id":0,"operations":[{"op":"scale","target":"all","factor":1.0}]}]}'


def _sd(scale=1.0):
    """A REAL model state_dict — the harness feeds these through delta_details and
    FedServer.load_state_dict, so the keys/shapes have to be the model's own."""
    from model.nidd_net import NiddNet
    return {k: v.detach().clone() * scale for k, v in NiddNet().state_dict().items()}


class _Ctx:
    """The three RoundContext fields _sample_attack touches."""

    def __init__(self, budget=2, n_pool=4, round_num=66):
        self.round_num = round_num
        self.budget = budget
        self.pool_benign = {cid: _sd(cid + 1) for cid in range(n_pool)}


class _Policy:
    """Returns the scripted texts in order, then repeats the last one forever."""

    def __init__(self, texts, completed=True):
        self.texts = list(texts)
        self.calls = []                       # (adapter, temperature) per generate
        self.last_generation_completed = [bool(completed)]

    def generate(self, adapter, system, user, n=1, temperature=0.0, max_new_tokens=512):
        self.calls.append((adapter, temperature))
        text = self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]
        return [text] * n


# --- _sample_attack ---------------------------------------------------------

def test_first_usable_action_is_taken_without_retrying():
    policy, ctx = _Policy([_GOOD]), _Ctx(budget=2)
    poisoned, chosen, malformed, attempts = _sample_attack(
        policy, AttackerAgent(), ctx, "sys", "usr", adapter="attacker",
        temperature=0.7, max_new_tokens=512, retries=3)
    assert chosen == [0, 1] and len(poisoned) == 2 and malformed == 0
    assert attempts == 1 and len(policy.calls) == 1          # no wasted generations
    assert policy.calls[0] == ("attacker", 0.7)              # caller's temperature honored


def test_unusable_action_is_resampled_and_the_round_survives():
    """The reported failure: one bad sample (truncated mid-JSON) killed a 100-round run."""
    policy, ctx = _Policy([_TRUNCATED, _TRUNCATED, _GOOD]), _Ctx(budget=2)
    poisoned, chosen, malformed, attempts = _sample_attack(
        policy, AttackerAgent(), ctx, "sys", "usr", adapter="attacker",
        temperature=0.7, max_new_tokens=512, retries=3)
    assert chosen == [0, 1] and len(poisoned) == 2 and malformed == 0
    assert attempts == 3 and len(policy.calls) == 3


def test_noop_plan_is_also_retried():
    """A plan that parses fine but changes no weight is unusable too (scale factor=1.0):
    select_and_apply refuses to label unchanged benign weights as poison."""
    policy, ctx = _Policy([_NOOP, _GOOD]), _Ctx(budget=2)
    _p, chosen, _m, attempts = _sample_attack(
        policy, AttackerAgent(), ctx, "sys", "usr", adapter="attacker",
        temperature=0.7, max_new_tokens=512, retries=2)
    assert chosen == [0, 1] and attempts == 2


def test_greedy_retries_are_sampled_so_they_are_not_identical_redraws():
    """At temperature 0 the model would re-emit the same unusable text verbatim, so
    retries (and only retries) are forced above 0."""
    policy, ctx = _Policy([_TRUNCATED, _GOOD]), _Ctx(budget=2)
    _sample_attack(policy, AttackerAgent(), ctx, "sys", "usr", adapter="attacker",
                   temperature=0.0, max_new_tokens=512, retries=3)
    assert policy.calls[0][1] == 0.0                          # first attempt: as asked
    assert policy.calls[1][1] >= harness._RETRY_TEMPERATURE_FLOOR


def test_exhausted_retries_report_the_last_failure_not_an_exception():
    policy, ctx = _Policy([_TRUNCATED]), _Ctx(budget=3)
    poisoned, chosen, malformed, attempts = _sample_attack(
        policy, AttackerAgent(), ctx, "sys", "usr", adapter="attacker",
        temperature=0.7, max_new_tokens=512, retries=2)
    assert poisoned == {} and chosen == [] and malformed == 3
    assert attempts == 3 and len(policy.calls) == 3


def test_zero_retries_means_one_attempt():
    policy, ctx = _Policy([_TRUNCATED, _GOOD]), _Ctx(budget=2)
    _p, chosen, _m, attempts = _sample_attack(
        policy, AttackerAgent(), ctx, "sys", "usr", adapter="attacker",
        temperature=0.7, max_new_tokens=512, retries=0)
    assert chosen == [] and attempts == 1 and len(policy.calls) == 1


def test_token_cap_is_reported_when_the_generation_was_cut_short():
    import io

    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    log = logging.getLogger("benchmark")
    log.setLevel(logging.INFO)
    log.addHandler(h)
    try:
        # completed=False -> the generation hit max_new_tokens instead of emitting EOS.
        _sample_attack(_Policy([_TRUNCATED], completed=False), AttackerAgent(),
                       _Ctx(budget=2), "sys", "usr", adapter="attacker",
                       temperature=0.7, max_new_tokens=512, retries=0)
        out = buf.getvalue()
        assert "token cap" in out and "rl.max_new_tokens" in out
        # ...and the offending output is quoted, so the failure is diagnosable.
        assert '{"clients"' in out
        buf.truncate(0), buf.seek(0)
        # A complete generation that simply parsed badly must NOT blame the cap.
        _sample_attack(_Policy([_TRUNCATED], completed=True), AttackerAgent(),
                       _Ctx(budget=2), "sys", "usr", adapter="attacker",
                       temperature=0.7, max_new_tokens=512, retries=0)
        assert "token cap" not in buf.getvalue()
    finally:
        log.removeHandler(h)


def test_hit_token_cap_tolerates_generators_without_the_flag():
    class _Bare:
        pass

    assert _hit_token_cap(_Bare()) is False
    assert _hit_token_cap(_Policy([_GOOD], completed=True)) is False
    assert _hit_token_cap(_Policy([_GOOD], completed=False)) is True


def test_snippet_is_one_capped_line():
    assert _snippet("a\n  b\tc") == "a b c"
    long = _snippet("x" * 500)
    assert long.startswith("x" * 240) and "500 chars total" in long
    assert len(long) < 300


# --- the harness round loop: skip, don't abort ------------------------------

class _Env:
    """Minimal round generator: a fixed pool, no model evolution (the benchmark
    never commits to the env — each defense carries its own global)."""

    def __init__(self, budget=2, n_clients=4, n_rounds_seen=0):
        self.budget = budget
        self.n_clients = n_clients
        self.round_num = n_rounds_seen
        self.global_weights = _sd()
        self.committed = []

    def begin_round(self):
        self.round_num += 1
        return _Ctx(budget=self.budget, n_pool=self.n_clients, round_num=self.round_num)

    def set_committed_poison(self, chosen_ids):
        self.committed.append(sorted(chosen_ids))

    def build_updates(self, poisoned_by_client):
        return [ModelUpdate(client_id=cid,
                            weights=poisoned_by_client.get(cid, _sd(cid + 1)),
                            metadata={"poisoned": cid in poisoned_by_client})
                for cid in range(self.n_clients)]


class _StepResult:
    def __init__(self, new_global, verdicts):
        self.new_global, self.verdicts = new_global, verdicts


class _Defense:
    """Flags exactly the ground-truth poisoners and keeps its initial model, so the
    metrics are fully determined by which rounds were measured."""

    def __init__(self):
        self.steps = 0
        self.global_ = None

    def reset(self, init_global):
        self.global_ = init_global

    def step(self, updates, poisoned_ids):
        self.steps += 1
        return _StepResult(self.global_, [
            DetectionVerdict(u.client_id, u.client_id in poisoned_ids, 1.0, "")
            for u in updates
        ])

    def global_weights(self):
        return self.global_


class _Loader:
    def __iter__(self):
        return iter(())


def _run(policy, n_rounds=3, retries=1, budget=2):
    """run_benchmark over a stub env/defense; returns (summary, defense)."""
    defense = _Defense()
    summaries, _metrics = run_benchmark(
        _Env(budget=budget), policy, AttackerAgent(), {"oracle": defense}, _Loader(),
        init_global=_sd(), baseline_accuracy=0.8, n_rounds=n_rounds,
        attack_temperature=0.7, max_new_tokens=512, attack_retries=retries,
        log_every=100)
    return summaries["oracle"], defense


def test_run_benchmark_no_longer_aborts_on_an_unusable_round():
    """Round 2 of 3 is unusable at every attempt: the run must finish, not raise."""
    texts = [_GOOD,                       # round 1
             _TRUNCATED, _TRUNCATED,      # round 2: attempt + 1 retry, both bad
             _GOOD, _GOOD]                # round 3
    summary, defense = _run(_Policy(texts), n_rounds=3, retries=1)
    # Only the two usable rounds were measured — the skipped one is excluded from
    # every defense's metrics rather than scored as an attack that never happened.
    assert summary["rounds"] == 2 and defense.steps == 2
    assert summary["malicious_total"] == 4 and summary["detection_rate"] == 1.0
    # mean_poisoned stays the exact quota, so the header's "exact quota" claim holds.
    assert summary["mean_poisoned"] == 2.0


def test_all_rounds_measured_when_every_action_is_usable():
    summary, defense = _run(_Policy([_GOOD]), n_rounds=4, retries=1)
    assert summary["rounds"] == 4 and defense.steps == 4


def test_a_wholly_broken_attacker_yields_an_empty_but_valid_report():
    """Nothing usable in any round: zero measured rounds, no crash, no fake poison."""
    summary, defense = _run(_Policy([_TRUNCATED]), n_rounds=3, retries=0)
    assert summary["rounds"] == 0 and defense.steps == 0
    assert summary["malicious_total"] == 0 and summary["mean_poisoned"] == 0.0
    # The report must still render this (the run's own summary of a failed attacker).
    from benchmark import report
    assert "oracle" in report.format_table([summary])


def _all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} benchmark-retry tests passed.")


if __name__ == "__main__":
    _all()
