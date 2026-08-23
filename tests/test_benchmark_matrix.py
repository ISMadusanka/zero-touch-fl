"""Tests for the attack x defense matrix: the harness invariants, the report and
the plot's history handling.

The invariants under test are the ones the whole comparison rests on — if any of
them silently broke, the benchmark would still produce a plausible-looking table
that meant something else:

* every attack in a round poisons the SAME clients (the trained attacker picks
  them and the baselines follow);
* every attack sees the SAME honest updates in a round;
* each attack gets its OWN defense instances, so one row cannot move another's
  global model;
* a round the LLM cannot produce a usable action for is dropped for the WHOLE
  panel, so the rows never cover different sets of rounds;
* the accuracy cache is a pure memo — identical results with it on and off.

Needs torch (the harness imports the FL server). Run:
    python tests/test_benchmark_matrix.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                              # noqa: E402

from agents.attacker_agent import AttackerAgent           # noqa: E402
from benchmark import plot, report                        # noqa: E402
from benchmark.attacks.base import Attack, DeltaAttack, broadcast  # noqa: E402
from benchmark.attacks.llm_attack import LLMAttack        # noqa: E402
from benchmark.harness import _AccuracyCache, run_attack_benchmark  # noqa: E402
from benchmark.metrics import DefenseMetrics              # noqa: E402
from core.types import DetectionVerdict, ModelUpdate      # noqa: E402

logging.getLogger("benchmark").addHandler(logging.NullHandler())
logging.getLogger("benchmark").propagate = False

_GOOD = ('{"clients":[{"id":0,"operations":[{"op":"scale","target":"all","factor":2.0}]},'
         '{"id":1,"operations":[{"op":"scale","target":"all","factor":0.5}]}]}')
_TRUNCATED = '{"clients":[{"id":0,"operations":[{"op":"scale","target":"all","fac'


_BASE_SD = None


def _sd(scale=1.0) -> dict:
    """A real model state_dict scaled by ``scale``.

    Deterministic on purpose: ``NiddNet()`` re-initialises randomly, so a helper that
    built a fresh one per call would make "the same bytes" untestable and would make
    two runs of the same panel disagree — which is exactly what the cache test has to
    rule out.
    """
    global _BASE_SD
    if _BASE_SD is None:
        from model.nidd_net import NiddNet
        torch.manual_seed(0)
        _BASE_SD = {k: v.detach().clone() for k, v in NiddNet().state_dict().items()}
    return {k: v.clone() * scale for k, v in _BASE_SD.items()}


class _Ctx:
    def __init__(self, budget, n_pool, round_num):
        self.round_num = round_num
        self.budget = budget
        self.pool_ids = list(range(n_pool))     # the compromised (controllable) clients
        self.pool_benign = {}
        self.goal = {}


class _Env:
    """Minimal round generator with real honest updates (the attacks are tensor code)."""

    def __init__(self, budget=2, n_clients=4, n_pool=None):
        self.budget = budget
        self.n_clients = n_clients
        self.n_pool = n_clients if n_pool is None else n_pool
        self.round_num = 0
        self.global_weights = _sd()
        self.committed = []
        self.honest_updates = []

    def begin_round(self):
        self.round_num += 1
        # New honest updates each round, so "every attack saw the same ones" is a real
        # assertion rather than a tautology over frozen tensors.
        self.honest_updates = [
            ModelUpdate(client_id=cid,
                        weights=_sd(1.0 + 0.01 * (cid + 1) * (self.round_num + 1)))
            for cid in range(self.n_clients)]
        ctx = _Ctx(self.budget, self.n_pool, self.round_num)
        ctx.pool_benign = {cid: self.honest_updates[cid].weights
                           for cid in range(self.n_pool)}
        return ctx

    def set_committed_poison(self, chosen_ids):
        self.committed.append(sorted(int(c) for c in chosen_ids))

    def build_updates(self, poisoned_by_client):
        return [ModelUpdate(client_id=cid,
                            weights=poisoned_by_client.get(cid,
                                                           self.honest_updates[cid].weights),
                            metadata={"poisoned": cid in poisoned_by_client})
                for cid in range(self.n_clients)]


class _Policy:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []
        self.last_generation_completed = [True]

    def generate(self, adapter, system, user, n=1, temperature=0.0, max_new_tokens=512):
        self.calls.append((adapter, temperature))
        return [self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]] * n


class _Recorder(DeltaAttack):
    """A baseline that records what it was shown, and shifts every client by a constant."""

    def __init__(self, name, shift=1.0):
        self.name = name
        self.shift = float(shift)
        self.seen = []

    def craft_deltas(self, ctx):
        self.seen.append({
            "round": ctx.round_num,
            "poisoned": list(ctx.poisoned_ids),
            "known": list(ctx.known_ids),
            "honest_digest": _AccuracyCache._digest(ctx.honest[0]),
            "reference": ctx.reference_accuracy,
        })
        mu = ctx.known_deltas().mean(0)
        return broadcast(ctx.poisoned_ids, mu + self.shift)


class _Defense:
    """Flags the ground truth and averages everything, so its global tracks the attack."""

    def __init__(self):
        self.steps = 0
        self.seen_ids = []
        self._global = None

    def reset(self, init_global):
        self._global = {k: v.clone() for k, v in init_global.items()}

    def step(self, updates, poisoned_ids):
        self.steps += 1
        self.seen_ids.append(sorted(poisoned_ids))
        self._global = {k: torch.stack([u.weights[k].float() for u in updates]).mean(0)
                        for k in updates[0].weights}
        return _StepResult(self._global, [
            DetectionVerdict(u.client_id, u.client_id in poisoned_ids, 1.0, "")
            for u in updates])

    def global_weights(self):
        return self._global


class _StepResult:
    def __init__(self, new_global, verdicts):
        self.new_global, self.verdicts = new_global, verdicts


class _Loader:
    def __iter__(self):
        return iter(())


def _run(attacks, n_rounds=3, budget=2, n_clients=4, n_pool=None, **kw):
    env = _Env(budget=budget, n_clients=n_clients, n_pool=n_pool)
    panels = {name: {"fedavg": _Defense(), "oracle": _Defense()} for name in attacks}
    summaries, metrics, info = run_attack_benchmark(
        env, attacks, panels, _Loader(), init_global=_sd(), baseline_accuracy=0.8,
        n_rounds=n_rounds, log_every=100, **kw)
    return env, panels, summaries, metrics, info


# --- the shared-round invariants -------------------------------------------

def test_baselines_poison_exactly_the_clients_the_llm_chose():
    llm = LLMAttack(_Policy([_GOOD]), AttackerAgent(), temperature=0.7)
    a, b = _Recorder("lie"), _Recorder("ipm", shift=2.0)
    env, _panels, _s, _m, _i = _run({"llm": llm, "lie": a, "ipm": b}, n_rounds=3)
    assert env.committed == [[0, 1]] * 3
    for rec in (a, b):
        assert [r["poisoned"] for r in rec.seen] == [[0, 1]] * 3


def test_every_attack_sees_the_same_honest_updates_in_a_round():
    a, b = _Recorder("lie"), _Recorder("ipm", shift=2.0)
    _run({"lie": a, "ipm": b}, n_rounds=3)
    assert [r["honest_digest"] for r in a.seen] == [r["honest_digest"] for r in b.seen]
    # ...and different ACROSS rounds, so the equality above is not vacuous.
    assert len(set(r["honest_digest"] for r in a.seen)) == 3


def test_knowledge_setting_controls_what_the_baselines_see():
    a = _Recorder("lie")
    _run({"lie": a}, n_rounds=1, budget=2, n_clients=6, n_pool=3, knowledge="partial")
    assert a.seen[0]["known"] == [0, 1, 2]                     # the compromised pool only
    b = _Recorder("lie")
    _run({"lie": b}, n_rounds=1, budget=2, n_clients=6, n_pool=3, knowledge="full")
    assert b.seen[0]["known"] == [0, 1, 2, 3, 4, 5]            # every client
    try:
        _run({"lie": _Recorder("lie")}, n_rounds=1, knowledge="omniscient")
    except ValueError:
        return
    raise AssertionError("an unknown knowledge setting must be rejected")


def test_without_an_llm_the_poisoned_set_is_the_pool_prefix():
    env, _p, _s, _m, _i = _run({"lie": _Recorder("lie")}, n_rounds=2, budget=3)
    assert env.committed == [[0, 1, 2]] * 2


def test_each_attack_gets_its_own_defense_instances():
    a, b = _Recorder("lie", shift=1.0), _Recorder("ipm", shift=50.0)
    _env, panels, _s, _m, _i = _run({"lie": a, "ipm": b}, n_rounds=2)
    assert panels["lie"]["fedavg"] is not panels["ipm"]["fedavg"]
    # Different attacks -> different aggregates. A shared panel would make these equal.
    ga, gb = panels["lie"]["fedavg"].global_weights(), panels["ipm"]["fedavg"].global_weights()
    assert not torch.allclose(ga["net.0.weight"], gb["net.0.weight"])


def test_each_attack_tracks_its_own_reference_accuracy():
    a, b = _Recorder("lie"), _Recorder("ipm", shift=50.0)
    _e, _p, _s, metrics, _i = _run({"lie": a, "ipm": b}, n_rounds=3)
    # Round 1 starts both at the clean baseline...
    assert a.seen[0]["reference"] == b.seen[0]["reference"] == 0.8
    # ...and from then on each observes the accuracy of ITS OWN undefended world, never
    # the damage some other row in the matrix did.
    for rec, name in ((a, "lie"), (b, "ipm")):
        undefended = [r["accuracy"] for r in metrics[name]["fedavg"].history]
        assert [r["reference"] for r in rec.seen][1:] == undefended[:-1]


def test_an_unusable_llm_round_is_dropped_for_the_whole_panel():
    llm = LLMAttack(_Policy([_GOOD, _TRUNCATED, _GOOD]), AttackerAgent(), retries=0)
    rec = _Recorder("lie")
    _env, panels, summaries, _m, info = _run({"llm": llm, "lie": rec}, n_rounds=3)
    assert info["measured_rounds"] == 2 and info["unusable_attack_rounds"] == 1
    # The baseline is measured over the SAME two rounds, not over all three.
    assert len(rec.seen) == 2 and [r["round"] for r in rec.seen] == [1, 3]
    assert summaries["llm"]["oracle"]["rounds"] == summaries["lie"]["oracle"]["rounds"] == 2
    assert panels["lie"]["oracle"].steps == 2


def test_an_attack_that_returns_the_wrong_clients_is_an_error():
    class _Wrong(Attack):
        name = "wrong"

        def craft(self, ctx):
            return {ctx.n_clients - 1: dict(ctx.global_weights)}

    try:
        _run({"wrong": _Wrong()}, n_rounds=1)
    except RuntimeError as e:
        assert "same clients" in str(e)
        return
    raise AssertionError("a mismatched poisoned set must not pass silently")


def test_at_most_one_llm_attack_per_panel():
    two = {"a": LLMAttack(_Policy([_GOOD]), AttackerAgent()),
           "b": LLMAttack(_Policy([_GOOD]), AttackerAgent())}
    try:
        _run(two, n_rounds=1)
    except ValueError:
        return
    raise AssertionError("two LLM attacks must be rejected")


# --- the accuracy cache -----------------------------------------------------

def test_accuracy_cache_is_a_pure_memo():
    class _Server:
        def __init__(self):
            self.evals = 0
            self.w = None

        def set_global_weights(self, w):
            self.w = w

        def evaluate(self, loader):
            self.evals += 1
            return float(self.w["net.0.weight"].sum())

    cache, server = _AccuracyCache(True), _Server()
    a = _sd(1.0)
    b = {k: v.clone() for k, v in a.items()}      # equal bytes, different objects
    assert cache.evaluate(server, a, None) == cache.evaluate(server, b, None)
    assert server.evals == 1 and cache.hits == 1
    assert cache.evaluate(server, _sd(2.0), None) != cache.evaluate(server, a, None)
    assert server.evals == 2

    off, server2 = _AccuracyCache(False), _Server()
    off.evaluate(server2, a, None); off.evaluate(server2, b, None)
    assert server2.evals == 2 and off.hits == 0


def test_matrix_results_are_identical_with_the_cache_off():
    def run(flag):
        _e, _p, summaries, _m, _i = _run(
            {"lie": _Recorder("lie"), "ipm": _Recorder("ipm", 3.0)},
            n_rounds=3, eval_cache=flag)
        return summaries

    on, off = run(True), run(False)
    for a in on:
        for d in on[a]:
            assert on[a][d] == off[a][d], (a, d)


# --- report + plot ----------------------------------------------------------

def _summaries():
    out = {}
    for attack, drop in (("llm", 0.20), ("lie", 0.05)):
        out[attack] = {}
        for defense in ("fedavg", "fltrust"):
            m = DefenseMetrics(defense, 0.8, target_drop=0.1, attack=attack)
            m.record(1, [DetectionVerdict(0, True, 1.0, "")], {0}, 0.8 - drop)
            out[attack][defense] = m.summary()
    return out


def test_matrix_report_marks_the_strongest_attack_per_defense():
    text = report.render_matrix(_summaries(), n_rounds=1, baseline_accuracy=0.8,
                                citations={"lie": "Baruch et al."},
                                run_info={"knowledge": "partial"})
    assert "ATTACK x DEFENSE BENCHMARK" in text and "PER-ATTACK DETAIL" in text
    assert "Baruch et al." in text and "partial" in text
    drops = text.split("ACC_DROP")[1].splitlines()
    llm_row = next(ln for ln in drops if ln.startswith("llm"))
    lie_row = next(ln for ln in drops if ln.startswith("lie"))
    assert "*" in llm_row and "*" not in lie_row      # llm did the most damage


def test_per_defense_table_gains_an_attack_column_only_when_needed():
    flat = [s for panel in _summaries().values() for s in panel.values()]
    assert report.format_table(flat).splitlines()[0].startswith("attack")
    one = list(_summaries()["llm"].values())
    assert report.format_table(one).splitlines()[0].startswith("defense")


def test_plot_detects_nested_and_flat_histories():
    flat = {"fedavg": [{"round": 1, "tp": 1, "fn": 0, "fp": 0, "tn": 3, "accuracy": 0.7}]}
    assert not plot.is_matrix_history(flat)
    assert plot.is_matrix_history({"llm": flat})
    assert plot.is_matrix_history({}) is False


def _all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} benchmark-matrix tests passed.")


if __name__ == "__main__":
    _all()
