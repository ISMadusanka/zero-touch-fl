"""The ``p_malicious`` calibration contract, asserted for every real defense.

``core.types.DetectionVerdict`` requires

    p_malicious >= 0.5   if and only if   is_suspicious

and the attacker's ``stealth`` reward (``1 - p`` averaged over the poisoned clients,
see ``rl.rewards.attacker_reward``) is built directly on that. Every producer here
violated it, in both directions, and the consequences were invisible in the logs
because a broken probability still looks like a number:

* FLTrust reported ``1 - ReLU(cos(delta_i, g0))``. Cosines between one client's delta
  and the root update are ~0.05 on a small model, so every client it ACCEPTED reported
  ``p ~ 0.95``. Stealth was ~0.03 whether the attack evaded detection or not — 3% of
  its configured weight, i.e. a dead term.
* DeFL reported ``votes/L`` against an ADAPTIVE threshold. On this codebase's
  two-layer model the rule settles at ``votes >= 1``, so a client it had just FLAGGED
  reported ``p = 1/2 = 0.5`` and the attacker still collected half the stealth bonus.
  One recorded round paid 0.44 — the highest attacker reward in the sample — for an
  attack that was fully detected and did no damage.
* Multi-Krum and DnC reported the cohort RANK. Bounded and monotone, but the mean is
  ~0.5 every round by construction, so ``p`` moved when OTHER clients moved and said
  nothing about whether this client was detected.

These tests run the actual defenses on constructed updates, so they fail if any future
change starts reporting a raw suspicion score in this field again.

Run:  python tests/test_p_malicious_calibration.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from benchmark.defenses.defl import DeFL  # noqa: E402
from benchmark.defenses.dnc import DnC  # noqa: E402
from benchmark.defenses.fltrust import FLTrust  # noqa: E402
from benchmark.defenses.multikrum import MultiKrum  # noqa: E402
from core.types import ModelUpdate  # noqa: E402
from data.feature_spec import DEFAULT_SPEC  # noqa: E402
from model.nidd_net import NiddNet  # noqa: E402
from rl.rewards import _soft_malicious_prob, attacker_reward  # noqa: E402

N_CLIENTS = 20
N_POISONED = 4
GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.02}


def _global():
    torch.manual_seed(0)
    return NiddNet().state_dict()


def _updates(gw, n=N_CLIENTS, n_pois=N_POISONED, scale=6.0):
    """``n`` clients around ``gw``; the first ``n_pois`` are scaled + sign-flipped."""
    torch.manual_seed(1)
    ups = []
    for cid in range(n):
        w = {k: v + 0.02 * torch.randn_like(v) for k, v in gw.items()}
        if cid < n_pois:
            w = {k: gw[k] - scale * (w[k] - gw[k]) for k in w}
        ups.append(ModelUpdate(cid, w, {"train_samples": 100, "poisoned": cid < n_pois}))
    return ups


def _defenses():
    torch.manual_seed(2)
    root = TensorDataset(torch.randn(100, DEFAULT_SPEC.input_dim),
                         torch.randint(0, DEFAULT_SPEC.n_classes, (100,)))
    return {
        "fltrust": FLTrust(DataLoader(root, batch_size=64, shuffle=True),
                           lr=0.002, local_epochs=5),
        "defl": DeFL(),
        "dnc": DnC(num_byzantine=5, sub_dim=10000),
        "multikrum": MultiKrum(num_byzantine=5),
    }


def _run(name, defense):
    gw = _global()
    defense.reset(gw)
    random.seed(0)
    return defense.step(_updates(gw), set()).verdicts


# --- the contract ------------------------------------------------------------

def test_every_defense_agrees_with_its_own_hard_flag():
    """p >= 0.5 exactly when the defense flagged the client. THE invariant."""
    for name, defense in _defenses().items():
        for v in _run(name, defense):
            assert v.p_malicious is not None, f"{name}: no p_malicious reported"
            assert 0.0 <= v.p_malicious <= 1.0, (name, v.client_id, v.p_malicious)
            assert v.is_suspicious == (v.p_malicious >= 0.5), (
                f"{name}: client {v.client_id} flagged={v.is_suspicious} but "
                f"p_malicious={v.p_malicious:.4f} — the calibration contract in "
                f"core.types.DetectionVerdict is broken"
            )


def test_stealth_never_pays_more_for_being_caught_than_for_evading():
    """The reward-level consequence of the invariant.

    Whatever a defense does internally, an attacker whose clients were ALL flagged
    must not out-score an otherwise identical attacker whose clients all evaded.
    Under ``votes/L`` (DeFL) it did.
    """
    for name, defense in _defenses().items():
        verdicts = _run(name, defense)
        poisoned = [v.client_id for v in verdicts if v.client_id < N_POISONED]
        caught = [v for v in verdicts if v.is_suspicious]
        evaded = [v for v in verdicts if not v.is_suspicious]
        if not caught or not evaded:
            continue                          # nothing to compare in this round
        # Same damage, same quota, same everything — only the verdicts differ.
        r_caught = attacker_reward(0.80, 0.79, GOAL, poisoned,
                                   [_relabel(v, cid) for cid, v in
                                    zip(poisoned, _cycle(caught, len(poisoned)))], 0)
        r_evaded = attacker_reward(0.80, 0.79, GOAL, poisoned,
                                   [_relabel(v, cid) for cid, v in
                                    zip(poisoned, _cycle(evaded, len(poisoned)))], 0)
        assert r_evaded > r_caught, (
            f"{name}: evading scored {r_evaded:.4f} but being caught scored "
            f"{r_caught:.4f} — the stealth gradient is inverted"
        )


def test_every_defense_keeps_a_usable_gradient_among_accepted_clients():
    """Stealth must stay CONTINUOUS on the accepted side.

    This is what the attacker can still improve: once a plan is flagged the reward
    can only tell it "caught", but among the plans that survived, a more honest-looking
    one must score better than one sitting on the boundary. A mean-based scale collapsed
    this (four extreme outliers pushed every accepted client to within 0.002 of 0.5),
    which is why the scale is a median.
    """
    for name, defense in _defenses().items():
        accepted = [v.p_malicious for v in _run(name, defense) if not v.is_suspicious]
        if len(accepted) < 3:
            continue
        assert len(set(round(p, 4) for p in accepted)) >= 3, (
            f"{name}: only {len(set(round(p, 4) for p in accepted))} distinct "
            f"p_malicious across {len(accepted)} accepted clients — stealth is "
            f"effectively constant, so GRPO gets no evasion gradient"
        )
        assert max(accepted) - min(accepted) > 0.01, (
            f"{name}: accepted clients span only "
            f"{max(accepted) - min(accepted):.5f} of p_malicious"
        )


def test_soft_prob_reads_the_calibrated_value_for_every_defense():
    """The reward's own accessor must agree with the verdict it was handed."""
    for name, defense in _defenses().items():
        for v in _run(name, defense):
            assert abs(_soft_malicious_prob(v) - v.p_malicious) < 1e-9, name


# --- helpers -----------------------------------------------------------------

def _cycle(items, n):
    return [items[i % len(items)] for i in range(n)]


def _relabel(v, client_id):
    """Same verdict, re-pointed at ``client_id`` so it lands on a poisoned client."""
    from core.types import DetectionVerdict
    return DetectionVerdict(client_id, v.is_suspicious, v.confidence, v.reason,
                            p_malicious=v.p_malicious)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} p_malicious calibration tests passed.")


if __name__ == "__main__":
    _run_all()
