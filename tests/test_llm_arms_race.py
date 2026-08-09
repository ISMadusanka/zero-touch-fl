"""End-to-end smoke test of the TWO-SIDED arms race (``defense.mode: llm``).

``tests/test_round_loop_integration.py`` covers the algorithmic path — one published
aggregator per round, attacker-only training. This one covers the other mode: the
defender is an LLM policy, both sides take turns learning, and the handoff is driven
by a three-consecutive-win streak (``rl.success_streak``).

No real LLM and no GPU: both agents are driven by scripted generators, so what is
under test is the WIRING — that ``AttackerTurn`` runs the frozen defender LLM and
FedAvgs the un-flagged clients, that ``DefenderTurn`` runs the frozen attacker and
scores verdicts, that the defender's prompt stays inside its context-fill band on a
real feature vector, and that ``PhaseController`` hands the phase back and forth on
three wins. Every one of those is a seam that only breaks when assembled.

    python tests/test_llm_arms_race.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from agents.attacker_agent import AttackerAgent  # noqa: E402
from agents.defender_agent import DefenderAgent  # noqa: E402
from model.mnist_net import MnistNet  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
from rl.switch import PhaseController, SwitchConfig, committed_success  # noqa: E402
from rl.turns import AttackerTurn, DefenderTurn  # noqa: E402

N_CLIENTS = 8
N_POISON = 3
RL_CFG = {"max_seq_len": 16384, "max_new_tokens": 1536, "max_context_fill": 0.5,
          "defender_max_new_tokens": 1024, "defender_max_context_fill": 0.60,
          "defender_max_prompt_fill": 0.30, "defender_min_prompt_fill": 0.20}


class ScriptedGen:
    """A turn generator that replays fixed text — stands in for a LoRA adapter.

    Records every (system, user) pair it is asked for, so a test can assert on the
    prompt the frozen opponent was actually shown.
    """

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system, user, n=1, temperature=0.0):
        self.calls.append((system, user))
        return [self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]] * n


def _loader(seed, n=96, batch=32):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.randn(n, 1, 28, 28, generator=g),
                      torch.randint(0, 10, (n,), generator=g)),
        batch_size=batch, shuffle=True)


def _env(begin=True):
    """A Phase-2 env with NO algorithmic defense — i.e. the defender LLM defends.

    ``begin`` opens the first round, which is what populates the honest updates a
    turn observes (``rl.schedule._step_round`` calls ``begin_round`` before building
    one). Pass ``False`` when the test drives the rounds itself.
    """
    cfg = {
        "fl": {"n_clients": N_CLIENTS, "device": "cpu", "training_rounds": 5,
               "benign_retrain_each_round": True, "n_compromisable": N_POISON,
               "lr": 0.01, "local_epochs": 1, "batch_size": 32},
        "attack": {"goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.02},
                   "max_poison_clients": N_POISON, "fixed_poison_clients": N_POISON,
                   "sample_budget_in_training": False},
        "defense": {"mode": "llm"},
    }
    torch.manual_seed(0)
    env = FLArmsRaceEnv(cfg, [_loader(i) for i in range(N_CLIENTS)], _loader(99, n=128),
                        random.Random(0), defense=None, curriculum=None)
    gw = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    cw = [{k: v + torch.randn_like(v) * 0.01 for k, v in gw.items()}
          for _ in range(N_CLIENTS)]
    env.reset(gw, cw, 0.5)
    if begin:
        env.begin_round()
    return env


def _agents():
    attacker = AttackerAgent({"fixed_poison_set": True, "rl": RL_CFG,
                              "attack_goal": {"type": "untargeted_degrade",
                                              "target_accuracy_drop": 0.02}})
    defender = DefenderAgent({"rl": RL_CFG})
    return attacker, defender


#: A plan that visibly enlarges each poisoned client's update.
ATTACK_TEXT = json.dumps({"clients": [
    {"id": cid, "operations": [{"op": "scale", "target": "all", "factor": 1.4}]}
    for cid in range(N_POISON)
]})


def _verdict_text(flagged):
    return json.dumps({"clients": [
        {"client_id": c, "is_suspicious": c in flagged, "confidence": 0.9}
        for c in range(N_CLIENTS)
    ]})


# --- the attacker's turn, defended by the LLM ------------------------------

def test_attacker_turn_runs_the_frozen_defender_llm_and_commits():
    env = _env()
    attacker, defender = _agents()
    # The frozen defender flags nothing, so every poisoned update reaches FedAvg.
    gen = ScriptedGen(_verdict_text(set()))
    turn = AttackerTurn(env, attacker, defender, defender_gen=gen,
                        reward_cfg={"alpha": 5.0, "beta": 0.1, "gamma": 0.5, "zeta": 0.05})
    assert not turn.algorithmic, "defense=None must select the defender-LLM path"

    reward = turn.reward(ATTACK_TEXT)
    assert isinstance(reward, float)
    info = turn.commit(ATTACK_TEXT)

    assert sorted(info["poisoned_ids"]) == list(range(N_POISON))
    assert [v.client_id for v in info["verdicts"]] == list(range(N_CLIENTS))
    assert info["n_malformed"] == 0
    assert 0.0 <= info["post_accuracy"] <= 1.0
    # The defender LLM was consulted for BOTH the scored rollout and the commit.
    assert len(gen.calls) == 2


def test_flagging_every_poisoned_client_beats_flagging_none():
    """The defense has to actually matter: dropping the poison must score better than
    letting it through. Without this the whole defender side is decorative."""
    env = _env()
    attacker, defender = _agents()
    caught = AttackerTurn(env, attacker, defender,
                          defender_gen=ScriptedGen(_verdict_text(set(range(N_POISON)))))
    missed = AttackerTurn(env, attacker, defender,
                          defender_gen=ScriptedGen(_verdict_text(set())))
    # Same clean counterfactual, same poison — only the verdicts differ.
    assert caught.reward(ATTACK_TEXT) < missed.reward(ATTACK_TEXT)


def test_the_defenders_prompt_stays_inside_its_context_band():
    """The prompt the frozen defender is shown must be the budgeted one — this is the
    seam where an unbudgeted agent would silently ship an oversized observation."""
    env = _env()
    attacker, defender = _agents()
    defender.bind_tokenizer(lambda s, u: (len(s) + len(u)) // 3)
    gen = ScriptedGen(_verdict_text({0}))
    AttackerTurn(env, attacker, defender, defender_gen=gen).commit(ATTACK_TEXT)

    st = defender.last_prompt_stats
    assert st["fits"] and st["prompt_fill"] <= 0.30 and st["fill"] <= 0.60
    _system, user = gen.calls[-1]
    payload = json.loads(user)
    assert len(payload["clients"]) == N_CLIENTS       # every client is described
    assert "cohort" in payload and "ranked" in payload


# --- the defender's turn, attacked by the frozen attacker ------------------

def test_defender_turn_scores_verdicts_against_the_frozen_attacker():
    env = _env()
    attacker, defender = _agents()
    turn = DefenderTurn(env, attacker, defender,
                        attacker_gen=ScriptedGen(ATTACK_TEXT),
                        reward_cfg={"mode": "soft_f1", "fpr_penalty": 1.0})

    assert sorted(turn.poisoned_ids) == list(range(N_POISON))
    perfect = turn.reward(_verdict_text(set(range(N_POISON))))
    blind = turn.reward(_verdict_text(set()))
    over = turn.reward(_verdict_text(set(range(N_CLIENTS))))
    assert perfect > blind, "catching the poison must beat missing it"
    assert perfect > over, "flagging everyone must not score like flagging correctly"

    info = turn.commit(_verdict_text(set(range(N_POISON))))
    assert [v.is_suspicious for v in info["verdicts"]][:N_POISON] == [True] * N_POISON


def test_defender_turn_refuses_to_run_against_an_algorithmic_defense():
    """A guard, not a nicety: silently training a defender policy whose verdicts are
    then ignored by an aggregator would waste a whole run."""
    env = _env()
    env.defense = object()          # pretend defense.mode: algorithmic
    attacker, defender = _agents()
    try:
        DefenderTurn(env, attacker, defender, attacker_gen=ScriptedGen(ATTACK_TEXT))
    except RuntimeError as e:
        assert "defense.mode" in str(e)
    else:
        raise AssertionError("DefenderTurn must refuse an algorithmic-defense env")


# --- the handoff ----------------------------------------------------------

def test_three_wins_hand_the_phase_over_and_back_on_real_verdicts():
    """Drive the controller with verdict objects produced by the real turns.

    The attacker wins while the frozen defender flags nothing; the defender wins once
    it catches every poisoned client. Three in a row either way flips the learner.

    The attacker's DAMAGE input is supplied explicitly rather than measured, because
    this env's model is randomly initialised on random labels — it already sits at
    chance accuracy, so there is no headroom for a reproducible drop and the measured
    value would make the test a coin flip. The damage bar itself is pinned precisely
    in ``tests/test_switch.py::test_attacker_relative_win_gate``; what is under test
    here is the EVASION half (real verdicts over a real poisoned set) and the handoff.
    """
    env = _env(begin=False)
    attacker, defender = _agents()
    cfg = SwitchConfig(min_phase_rounds=3, max_phase_rounds=200, success_streak=3,
                       attacker_min_drop=0.02, attacker_min_evaded=1.0,
                       defender_min_tpr=0.99, defender_max_fpr=0.10)
    bar = cfg.win_fraction * 0.02          # this round's relative damage bar
    ctrl = PhaseController(cfg, first_learner="attacker",
                           learners=("attacker", "defender"))

    # --- attacker phase: three passes through a defender that flags nothing ---
    for _round in range(3):
        env.begin_round()
        turn = AttackerTurn(env, attacker, defender,
                            defender_gen=ScriptedGen(_verdict_text(set())))
        info = turn.commit(ATTACK_TEXT)
        flagged = {v.client_id for v in info["verdicts"] if v.is_suspicious}
        assert not flagged & set(info["poisoned_ids"]), "every poisoner should evade here"

        args = (info["verdicts"], info["poisoned_ids"], cfg, turn.goal)
        assert committed_success("attacker", bar + 0.01, *args)
        # ...and the damage half of the gate is genuinely consulted: full evasion
        # alone is not a pass.
        assert not committed_success("attacker", bar - 0.01, *args)
        switch, reason = ctrl.record(True)
    assert (switch, reason) == (True, "success")
    ctrl.next_phase(reason)
    assert ctrl.learner == "defender"

    # --- defender phase: three rounds catching that frozen attacker exactly ---
    for _round in range(3):
        env.begin_round()
        turn = DefenderTurn(env, attacker, defender,
                            attacker_gen=ScriptedGen(ATTACK_TEXT))
        info = turn.commit(_verdict_text(set(turn.poisoned_ids)))
        won = committed_success("defender", 0.0, info["verdicts"],
                                info["poisoned_ids"], cfg)
        assert won, "perfect detection with no false positives must count as a win"
        switch, reason = ctrl.record(won)
    assert (switch, reason) == (True, "success")
    ctrl.next_phase(reason)
    assert ctrl.learner == "attacker" and ctrl.phase_index == 2


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} LLM arms-race tests passed.")


if __name__ == "__main__":
    _run()
