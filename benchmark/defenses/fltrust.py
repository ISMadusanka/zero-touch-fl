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
                 trust_flag_threshold: float = 0.0):
        super().__init__(device)
        self.eta = float(eta)
        self.trust_flag_threshold = float(trust_flag_threshold)
        # FLTrust holds its own model (to fine-tune on the root set each round).
        self.server = FedServer(device=device)
        self.root_client = BenignClient(client_id=-1, data_loader=root_loader,
                                        lr=lr, local_epochs=local_epochs, device=device)

    def reset(self, init_global):
        super().reset(init_global)               # clones into self._global
        self.server.set_global_weights(self._global)

    def step(self, updates, poisoned_ids) -> StepResult:
        gw = self._global
        keys = list(gw.keys())
        gw_flat = _flatten(gw, keys)

        # Server reference update g0: one local epoch on the clean root set,
        # starting from the CURRENT global model.
        self.server.set_global_weights(gw)
        root_update = self.root_client.train(self.server.model)   # absolute weights
        g0 = _flatten(root_update.weights, keys) - gw_flat

        deltas = [_flatten(u.weights, keys) - gw_flat for u in updates]
        agg, trust = fltrust_combine(deltas, g0)

        verdicts = [
            DetectionVerdict(u.client_id, ts <= self.trust_flag_threshold,
                             float(max(0.0, 1.0 - ts)), f"trust={ts:.3f}")
            for u, ts in zip(updates, trust)
        ]

        if agg is not None:
            new_flat = gw_flat + self.eta * agg              # w <- w + eta*g (Eq. 4/5)
            new_global = _unflatten(new_flat, gw, keys)
            self._global = new_global
        else:
            new_global = None    # no trusted client this round -> keep the global (guard /0)

        return StepResult(new_global, verdicts,
                          info={"g0_norm": float(g0.norm().item()), "trust_sum": sum(trust)})
