# Data Partitioning (`data/mnist_loader.py`)

Before any training begins, the central server downloads the standard MNIST
training dataset (60,000 images) and test dataset (10,000 images).

## Non-IID split — the FLTrust scheme (default)

With `data.iid: false` the 60,000 training images are partitioned across the
**20 clients** using the method from **FLTrust** (Cao et al., *"FLTrust:
Byzantine-robust Federated Learning via Trust Bootstrapping"*, NDSS 2021),
implemented in `partition_noniid_fltrust`:

1. **Groups.** The clients are split into `M = 10` groups (one per digit class) —
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

Each round the server sends the current global weights to every client; each client
loads them into a local copy, runs `local_epochs` of SGD on its own shard (batch
size 64, cross-entropy loss), and returns a `ModelUpdate` (updated weights + train
accuracy/loss/sample count) — never its raw data. The server FedAvgs the accepted
updates into the next global model. In Phase 1 all clients are honest, so no update
is filtered.

In Phase 2 each client trains on a **fresh slice of its own shard** per round
(`data/round_sampler.py`, `fl.client_round_fraction`, default 0.25 ≈ 750 examples).
That is what makes consecutive rounds differ when the global model is frozen: the
starting point is fixed, so without new local data every round would reproduce the
same honest updates. A slice always comes from the client's OWN shard, so the
non-IID skew above is preserved.

## Poisoned clients (`clients/malicious_client.py`)

A poisoned client runs **exactly the same procedure** — same examples, same batch
size, same epochs, same learning rate — on a dataset whose labels have been flipped
symmetrically (`y → 9 − y`, see `data/label_flip.py`). Nothing edits the weights
afterwards, so the update it submits is a genuine SGD trajectory of a real (if
wrong) objective.

How many of its labels are flipped is set per round by the attack ladder
(`agents/label_flip_attacker.py`): it starts at 100% of the client's round data and
backs off a notch every time the defense catches it, resetting to 100% once it
bottoms out at 50%. Because the fraction is applied to each client's **own**
per-round sample count, an unequal non-IID partition still flips the same
*proportion* everywhere.

`train_accuracy` in a poisoned client's metadata is measured against its **flipped**
labels, so it is that client's fit to its poisoned objective — expect it to be high
even though the model is being pushed away from the real task. The metadata also
carries `n_flipped`, `n_local_samples` and `flip_fraction`.

## Threat model note

Only the clients listed in **`attack.poison_client_ids`** (default `[0]` — a single
insider) flip labels; every other client is always honest. Keep that set below half
of `fl.n_clients` and the honest majority the defender's robust statistics rely on
always holds. Once it reaches half, that assumption is gone and the robust
aggregators (Multi-Krum, DnC, FLTrust's trust scores) are outside the regime they
are proved in — a legitimate experiment (it is the standard "attack success vs
fraction malicious" sweep), just one to read with the caveat. It is logged at
startup.
