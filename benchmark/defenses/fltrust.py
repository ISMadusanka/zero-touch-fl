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
from data.datasets import DEFAULT_DATASET
from server.fed_server import FedServer

from benchmark.defenses.base import Defense, StepResult


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
    ``(agg, trust_scores)`` where:

      trust_i = ReLU( cos(delta_i, g0) )                             (Eq. 2)
      agg     = (1 / sum_j trust_j) * sum_i trust_i * (||g0||/||delta_i||) delta_i  (Eq. 3/4)

    ``agg`` is None when every trust score is 0 (guard against /0). Kept separate
    from ``FLTrust.step`` so the math is unit-testable in isolation.
    """
    g0_norm = g0.norm()
    trust: list[float] = []
    agg = torch.zeros_like(g0)
    ts_sum = 0.0
    for di in deltas:
        di_norm = di.norm()
        if di_norm.item() <= 0.0 or g0_norm.item() <= 0.0:
            ts = 0.0
        else:
            cos = torch.dot(di, g0) / (di_norm * g0_norm)
            ts = float(torch.relu(cos).item())
        trust.append(ts)
        if ts > 0.0 and di_norm.item() > 0.0:
            agg = agg + ts * (g0_norm / di_norm) * di      # normalize to ||g0|| then trust-weight
            ts_sum += ts
    if ts_sum > 0.0:
        return agg / ts_sum, trust
    return None, trust


class FLTrust(Defense):
    name = "fltrust"

    def __init__(self, root_loader, lr: float, local_epochs: int = 1,
                 device: str = "cpu", eta: float = 1.0,
                 trust_flag_threshold: float = 0.0,
                 dataset: str = DEFAULT_DATASET):
        super().__init__(device)
        self.eta = float(eta)
        self.trust_flag_threshold = float(trust_flag_threshold)
        # FLTrust holds its own model (to fine-tune on the root set each round),
        # so it is the one defense that must know the dataset: its architecture
        # has to match the clients' and the root loader's data.
        self.server = FedServer(device=device, dataset=dataset)
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
        agg, trust = fltrust_combine(deltas, g0)

        # Trust is ReLU(cos) in [0, 1], so `1 - ts` is already a calibrated
        # P(malicious): 1.0 for a client pointing away from the trusted direction
        # (which is also exactly the drop condition), falling continuously to 0 for
        # a client perfectly aligned with it. It goes in ``p_malicious``; the
        # ``confidence`` slot means certainty in the VERDICT (see core.types), which
        # for a threshold at trust<=0 is |2p - 1|. Putting the suspicion score in
        # ``confidence`` inverted the attacker's stealth reward — see
        # ``rl.rewards._soft_malicious_prob``.
        verdicts = []
        for u, ts in zip(updates, trust):
            p_mal = max(0.0, min(1.0, 1.0 - ts))
            verdicts.append(DetectionVerdict(
                u.client_id, ts <= self.trust_flag_threshold,
                abs(2.0 * p_mal - 1.0), f"trust={ts:.3f}",
                p_malicious=p_mal,
            ))

        if agg is not None:
            new_flat = gw_flat + self.eta * agg              # w <- w + eta*g (Eq. 4/5)
            new_global = _unflatten(new_flat, gw, keys)
            self._global = new_global
        else:
            new_global = None    # no trusted client this round -> keep the global (guard /0)

        return StepResult(new_global, verdicts,
                          info={"g0_norm": float(g0.norm().item()), "trust_sum": sum(trust)})
