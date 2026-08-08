"""FLTrust (Cao et al., "FLTrust: Byzantine-robust Federated Learning via Trust
Bootstrapping", NDSS 2021) as a benchmark defense.

Per round the server fine-tunes the CURRENT global model on a small CLEAN root
dataset to get a trusted reference update g0. Each client update is turned into a
delta vs the current global; its trust score is

    TS_i = ReLU( cos(delta_i, g0) )                         (paper Eq. 2)

so a client pointing AWAY from the trusted direction gets zero trust and is
dropped. Surviving deltas are rescaled to the server update's magnitude

    delta_bar_i = (||g0|| / ||delta_i||) * delta_i          (paper Eq. 3)

(this neutralises magnitude-scaling attacks), then combined as a trust-weighted
average and applied to the global model:

    g = (1 / sum_j TS_j) * sum_i TS_i * delta_bar_i         (paper Eq. 4)
    w <- w + eta * g                                         (paper Eq. 5)

Sign convention: client submissions in this codebase are ABSOLUTE weights, so we
form deltas delta_i = w_i - w_global and g0 = w_root - w_global (w_root = the
global after one local epoch on the root set). FLTrust never needs to know how
many clients are malicious.

Detection read-out: a client is reported as REJECTED (is_suspicious=True) when its
trust score is <= ``trust_flag_threshold`` (default 0.0 — i.e. ReLU zeroed it, the
client points away from the trusted update). With the default threshold this set
is EXACTLY the clients excluded from the aggregate (kept iff trust > 0), so the
detection read-out and the aggregation agree. This is a DERIVED proxy for a binary
flag (FLTrust really assigns continuous trust), so compare it to an explicit
classifier loosely — see benchmark/README.md.
"""
import torch

from core.types import DetectionVerdict
from clients.benign_client import BenignClient
from server.fed_server import FedServer

from benchmark.defenses.base import Defense, StepResult, boundary_calibrated_p


def _flatten(state_dict: dict, keys: list) -> "torch.Tensor":
    """Concatenate the given keys into one float vector (fixed key order)."""
    return torch.cat([state_dict[k].reshape(-1).float() for k in keys])


def _unflatten(flat: "torch.Tensor", ref: dict, keys: list) -> dict:
    out, i = {}, 0
    for k in keys:
        n = ref[k].numel()
        out[k] = flat[i:i + n].reshape(ref[k].shape).to(ref[k].dtype)
        i += n
    return out


def fltrust_combine(deltas: list, g0: "torch.Tensor"):
    """Pure FLTrust aggregation on flattened vectors (no model/data needed).

    ``deltas`` = list of per-client deltas (client weights minus current global,
    flattened); ``g0`` = the server root update (flattened). Returns
    ``(agg, trust_scores, cosines)`` where:

      cos_i   = cos(delta_i, g0)                                     in [-1, 1]
      trust_i = ReLU( cos_i )                                        (Eq. 2)
      agg     = (1 / sum_j trust_j) * sum_i trust_i * (||g0||/||delta_i||) delta_i  (Eq. 3/4)

    ``agg`` is None when every trust score is 0 (guard against /0). Kept separate
    from ``FLTrust.step`` so the math is unit-testable in isolation.

    The raw ``cosines`` are returned alongside ``trust`` because ReLU DESTROYS the
    only information that distinguishes rejected clients from each other: every
    client pointing away from the trusted direction — whether slightly or exactly
    opposite — collapses to ``trust = 0``. The aggregation only needs the ReLU'd
    value, but the calibrated ``p_malicious`` needs the signed margin, so the
    verdicts are built from the cosine (see :meth:`FLTrust.step`).
    """
    g0_norm = g0.norm()
    trust: list[float] = []
    cosines: list[float] = []
    agg = torch.zeros_like(g0)
    ts_sum = 0.0
    for di in deltas:
        di_norm = di.norm()
        if di_norm.item() <= 0.0 or g0_norm.item() <= 0.0:
            # A zero-length delta (or a zero root update) has no direction at all;
            # it is neither aligned nor opposed, so the boundary value is 0.
            cos = 0.0
            ts = 0.0
        else:
            cos = float((torch.dot(di, g0) / (di_norm * g0_norm)).item())
            ts = max(0.0, cos)
        cosines.append(cos)
        trust.append(ts)
        if ts > 0.0 and di_norm.item() > 0.0:
            agg = agg + ts * (g0_norm / di_norm) * di      # normalize to ||g0|| then trust-weight
            ts_sum += ts
    if ts_sum > 0.0:
        return agg / ts_sum, trust, cosines
    return None, trust, cosines


class FLTrust(Defense):
    name = "fltrust"

    def __init__(self, root_loader, lr: float, local_epochs: int = 1,
                 device: str = "cpu", eta: float = 1.0,
                 trust_flag_threshold: float = 0.0):
        super().__init__(device)
        self.eta = float(eta)
        self.trust_flag_threshold = float(trust_flag_threshold)
        # FLTrust holds its own model (to fine-tune on the root set each round).
        self.server = FedServer(device=device)
        self.root_client = BenignClient(client_id=-1, data_loader=root_loader,
                                        lr=lr, local_epochs=local_epochs, device=device)
        self._g0_cache: tuple[bytes, "torch.Tensor"] | None = None

    def reset(self, init_global):
        super().reset(init_global)               # clones into self._global
        self.server.set_global_weights(self._global)
        self._g0_cache = None

    def _root_update(self, gw: dict, gw_flat: "torch.Tensor", keys: list) -> "torch.Tensor":
        """The trusted reference direction ``g0 = w_root - w_global``, computed ONCE
        per distinct global model and cached.

        The caching is a correctness requirement, not an optimization. ``g0`` comes
        from SGD over a shuffled root loader, so it is a random variable: re-running
        it per call gave every scored rollout in a GRPO group a DIFFERENT reference
        direction. Since trust is ``ReLU(cos(delta_i, g0))`` and Eq. 3 rescales every
        accepted delta to ``||g0||``, both the verdicts and the post-aggregation
        accuracy then moved between rollouts for reasons that had nothing to do with
        the attacker's action — and GRPO's advantage is exactly the within-group
        reward spread, so it was fitting root-training noise. It also broke the
        contract ``AlgorithmicDefender.run(commit=False)`` advertises ("repeated
        scoring calls within a round are independent and identically defended") and
        meant the COMMITTED round was graded by a different defense than the reward
        the policy trained on.

        Keying on the global's contents (rather than a round counter) needs no new
        interface: within a round every ``step`` sees the byte-identical global that
        ``sync_global`` installed, so all G rollouts, the clean counterfactual and
        the commit share one ``g0``; the next round's global differs, so the cache
        misses and the reference is legitimately redrawn.
        """
        import hashlib
        digest = hashlib.blake2b(
            gw_flat.detach().to("cpu").contiguous().numpy().tobytes(),
            digest_size=16,
        )
        digest.update("\0".join(keys).encode())   # guard against a key-order change
        key = digest.digest()
        if self._g0_cache is not None and self._g0_cache[0] == key:
            return self._g0_cache[1]

        self.server.set_global_weights(gw)
        root_update = self.root_client.train(self.server.model)   # absolute weights
        g0 = _flatten(root_update.weights, keys) - gw_flat
        self._g0_cache = (key, g0)
        return g0

    def step(self, updates, poisoned_ids) -> StepResult:
        gw = self._global
        keys = list(gw.keys())
        gw_flat = _flatten(gw, keys)

        # Server reference update g0: root_epochs local epochs on the clean root
        # set, starting from the CURRENT global model. Cached per global — see
        # _root_update for why that is required, not just faster.
        g0 = self._root_update(gw, gw_flat, keys)

        deltas = [_flatten(u.weights, keys) - gw_flat for u in updates]
        agg, trust, cosines = fltrust_combine(deltas, g0)

        # The verdict boundary is ``trust <= trust_flag_threshold``. Because trust is
        # ReLU(cos) and the threshold is >= 0, that is exactly ``cos <= threshold``,
        # so the cosine is the score to calibrate against — and unlike trust it keeps
        # a signed margin for the rejected side.
        #
        # ``1 - trust`` was NOT a calibrated P(malicious), which is the bug this
        # replaces: in a small model the cosine between any single client's delta and
        # the root update is ~0.05, so every client FLTrust happily accepted reported
        # ``p ~ 0.95``. The attacker's stealth term (``1 - p``) was then ~0.03 whether
        # it evaded detection or not — a dead signal worth 3% of its configured
        # weight. ``boundary_calibrated_p`` puts 0.5 exactly on the drop condition, so
        # accepted clients land below it and rejected ones above.
        #
        # ``confidence`` means certainty in the VERDICT (see core.types) = |2p - 1|.
        flags = [ts <= self.trust_flag_threshold for ts in trust]
        cos_threshold = max(0.0, self.trust_flag_threshold)
        p_mals = boundary_calibrated_p(cosines, cos_threshold,
                                       higher_is_suspicious=False, flags=flags)
        verdicts = [
            DetectionVerdict(
                u.client_id, flags[i], abs(2.0 * p_mals[i] - 1.0),
                f"trust={trust[i]:.3f} cos={cosines[i]:+.3f}",
                p_malicious=p_mals[i],
            )
            for i, u in enumerate(updates)
        ]

        if agg is not None:
            new_flat = gw_flat + self.eta * agg              # w <- w + eta*g (Eq. 4/5)
            new_global = _unflatten(new_flat, gw, keys)
            self._global = new_global
        else:
            new_global = None    # no trusted client this round -> keep the global (guard /0)

        return StepResult(new_global, verdicts,
                          info={"g0_norm": float(g0.norm().item()),
                                "trust_sum": sum(trust),
                                "n_accepted": sum(1 for f in flags if not f)})
