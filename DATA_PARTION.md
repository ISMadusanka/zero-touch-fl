# Data Partitioning (`data/nidd_loader.py`)

The FL task is **5G-NIDD** (Samarakoon et al., 2022): 1,215,890 labelled network
flows captured on the University of Oulu 5G Test Network — benign traffic plus
eight attacks (UDPFlood, HTTPFlood, SlowrateDoS, TCPConnectScan, SYNScan, UDPScan,
SYNFlood, ICMPFlood) described by ~52 Argus flow features. Nine classes counting
benign.

The dataset is too large to vendor, so it is **downloaded on first use** rather
than placed by hand. `data.source: kaggle` (the default) fetches the public,
CC BY 4.0 mirror `humera11/5g-nidd-dataset` with `kagglehub` — no Kaggle account
or API token required — and caches it under `~/.cache/kagglehub`. The download is
reached only on a preprocessing cache *miss*, so it costs one transfer per machine.

`data.source: csv` still reads a local copy at `data.csv_path` (a combined CSV, or
a directory of the per-attack CSVs — found recursively and concatenated), which is
what an air-gapped box or the
[IEEE DataPort](https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless)
release wants. To exercise the pipeline without the data at all, set
`data.source: synthetic`; that generates 5G-NIDD-shaped traffic through the same
preprocessing path and logs a warning on every run, because no number produced
from it means anything.

## Preprocessing (before any partitioning)

Order matters here, and it is deliberate:

1. **Drop identifier and leakage columns.** `Attack Tool` is a restatement of the
   label; `SrcAddr`, `DstAddr`, `Sport`, `Dport`, `StartTime`, `Seq` and friends
   are identifiers. Dropping them is load-bearing, not tidiness: 5G-NIDD's attack
   traffic originates from a handful of testbed hosts, so a model given the source
   address scores ~100% by memorising *"this IP is the attacker"* and learns
   nothing about traffic. Since the attacker's reward **is** test accuracy, that
   would make the whole arms race measure address lookup. The full list is
   `data/nidd_loader.py:DEFAULT_DROP_COLUMNS`; add more with `data.drop_columns`.
2. **Subsample** to `data.max_samples` (default 100,000), stratified. The full 1.2M
   flows over 20 clients would be ~60k examples per client per round — a different
   experiment, not a slower one.
3. **Split** train/test (`data.test_fraction`, default 0.2), stratified and seeded.
4. **Fit preprocessing on the train split only** — categorical vocabularies,
   missing-value medians, standardization statistics, and the top-K feature
   selector — then transform both splits.

Steps 3-then-4 are in that order because fitting the scaler or the selector on all
rows and splitting afterwards leaks test-set statistics into the model, which on an
intrusion-detection dataset inflates accuracy enough to matter.

`data.n_features` (default 32) picks the K most class-separating columns by one-way
**ANOVA F statistic** computed on the train split, ties broken by column name so
the selection is identical on every machine. That is what makes the model's
parameter count a config constant rather than a property of whichever CSV was
handed over.

Results are cached under `data.cache_dir`, keyed by the options plus the CSV's
size/mtime, so the ~1.2M-row parse happens once rather than on every run,
benchmark and resume. The resulting shape is published to `schema.json` and
consumed by `model.build_model()` — see `data/feature_spec.py`.

## Non-IID split — the FLTrust scheme (default)

With `data.iid: false` the training flows are partitioned across the **20 clients**
using the method from **FLTrust** (Cao et al., *"FLTrust: Byzantine-robust
Federated Learning via Trust Bootstrapping"*, NDSS 2021), implemented in
`partition_noniid_fltrust`:

1. **Groups.** The clients are split into `M = 9` groups (one per attack class,
   benign included) — with 20 clients that is **2-3 clients per group**, assigned
   round-robin. The class count comes from the data, so a binary run
   (`data.label_mode: binary`) forms 2 groups instead.
2. **Biased routing.** A flow with label `l` is assigned to **group `l` with
   probability `q`** (the *bias probability*, `data.noniid_bias`, default **0.5**),
   and to any *other* group with probability `(1 − q)/(M − 1)`.
3. **Within a group.** The flows routed to a group are split **evenly and at
   random** across that group's clients.

`q = 1/M ≈ 0.111` reproduces an **IID** split; larger `q` makes each group (and its
clients) increasingly dominated by its own attack class. At the default `q = 0.5`
the 20 clients hold **genuinely different label distributions** — which is why
*which* client the attacker compromises matters.

The partition is **seeded** (`fl.poison_seed`) so runs are reproducible, and it is
disjoint and covers every flow (no sample is dropped or duplicated).

### Shard sizes are unequal here, unlike on MNIST

MNIST's ten classes are near-uniform, so its groups came out the same size.
5G-NIDD's are not: Benign is ~39% of flows, UDPFlood ~38%, and ICMPFlood ~0.09%.
The groups attracting the two dominant classes therefore hold far more examples.
At the shipped settings that is a measured **3.6x spread** across the 20 clients —
2,510 / 2,946 / 9,078 examples at min / median / max — which is much less than the
400x the raw class ratio suggests, because at `q = 0.5` half of every class is
spread uniformly over the other groups and that floors the rare-class groups.
Raising `q` toward 1.0 removes the floor and the spread grows sharply.

Two consequences worth knowing:

* `RoundDataSampler` computes its per-round slice **per client**, not once for the
  federation, so `fl.client_round_fraction: 0.25` means 628 to 2,270 examples
  depending on the client rather than one number.
* At `data.class_balance: natural` (the default, faithful to the dataset) the
  rarest class has ~90 rows at `max_samples: 100000`, so the model will effectively
  not learn it. That is a property of 5G-NIDD, not a bug. Set
  `data.class_balance: balanced` to cap every class equally instead, at the cost of
  no longer being the published distribution.

## IID split (optional)

With `data.iid: true`, `partition_iid` shuffles all training indices and gives each
client an equal shard (4,000 flows each for 20 clients at the default
`max_samples`), so every client sees a roughly uniform mix of all nine classes.

## Client training (`clients/benign_client.py`)

Each round the server sends the current global weights to every client; each
client loads them into a local copy, runs `local_epochs` of SGD on its own shard
(batch size 64, cross-entropy loss), and returns a `ModelUpdate` (updated weights
+ train accuracy/loss/sample count) — never its raw data. The server FedAvgs the
accepted updates into the next global model. In Phase 1 all clients are honest, so
no update is filtered.

Clients receive `(batch, 32)` feature tensors — the model is a plain
fully-connected network, not a CNN, because averaging neighbouring *columns* of an
Argus record ("source TTL" next to "destination TTL") is meaningless. See
`model/nidd_net.py`.

## Threat model note

Only the **first `fl.n_compromisable` clients** are reachable by the attacker; the
remaining clients are always honest. Note the shipped config sets
`attack.fixed_poison_clients: 10` of `fl.n_clients: 20`, which is *at* the boundary
rather than below it — the robust statistics Multi-Krum / DnC / FLTrust rely on are
proved under a strict honest majority, so raise `fl.n_clients` to restore one.
