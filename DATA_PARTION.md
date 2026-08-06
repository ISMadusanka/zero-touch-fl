# Data Partitioning (`data/loaders.py`)

Before any training begins, the central server downloads the dataset selected by
`--dataset` (see [`data/datasets.py`](data/datasets.py) for the registry):

| `--dataset` | Train / test | Input |
|---|---|---|
| `mnist` (default) | 60,000 / 10,000 | 1×28×28 grayscale digits |
| `cifar10` | 50,000 / 10,000 | 3×32×32 colour images |

Both have **10 classes**, and the partitioning below reads only the label list —
so it is *identical* for either dataset. Everything that follows says "images" and
"digit class" from the MNIST default; substitute "object class" for CIFAR-10.

## Non-IID split — the FLTrust scheme (default)

With `data.iid: false` the training images are partitioned across the
**20 clients** using the method from **FLTrust** (Cao et al., *"FLTrust:
Byzantine-robust Federated Learning via Trust Bootstrapping"*, NDSS 2021),
implemented in `partition_noniid_fltrust`:

1. **Groups.** The clients are split into `M = 10` groups (one per class) —
   with 20 clients that is **2 clients per group** (client `g` and client `g+10`
   form group `g`).
2. **Biased routing.** A training image with label `l` is assigned to **group `l`
   with probability `q`** (the *bias probability*, `data.noniid_bias`, default
   **0.5**), and to any *other* group with probability `(1 − q)/(M − 1)`.
3. **Within a group.** The images routed to a group are split **evenly and at
   random** across that group's clients.

`q = 1/M = 0.1` reproduces an **IID** split; larger `q` makes each group (and its
clients) increasingly dominated by its own digit class. At the default `q = 0.5`,
about half of each digit's images concentrate in that digit's group, so the 20
clients hold **genuinely different label distributions** — which is why *which*
client the attacker compromises now matters.

The partition is **seeded** (`fl.poison_seed`) so runs are reproducible, and it is
disjoint and covers every image (no sample is dropped or duplicated).

## IID split (optional)

With `data.iid: true`, `partition_iid` shuffles all 60,000 indices and gives each
client an equal shard (3,000 images each for 20 clients), so every client sees a
roughly uniform mix of all ten digits.

## Client training (`clients/benign_client.py`)

Each round the server sends the current global weights to every client; each
client loads them into a local copy, runs `local_epochs` of SGD on its own shard
(batch size 64, cross-entropy loss), and returns a `ModelUpdate` (updated weights
+ train accuracy/loss/sample count) — never its raw data. The server FedAvgs the
accepted updates into the next global model. In Phase 1 all clients are honest, so
no update is filtered.

## Threat model note

Only the **first `fl.n_compromisable` clients** (default 5: ids `0..4`) are
reachable by the attacker; the remaining 15 are always honest. Because ≤ 5 of 20
clients can ever be poisoned, the honest majority the defender's robust
statistics rely on always holds.
