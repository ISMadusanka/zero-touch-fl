# Codebase Structure

**Analysis Date:** 2026-08-01

## Directory Layout

```
zero-touch-fl/
├── agents/                    # LLM agents: attacker selection + plans, defender classification
│   ├── __init__.py
│   ├── attacker_agent.py      # AttackerAgent: prompt builder + plan parser/applier
│   ├── attack_ops.py          # 11 attack operators: scale_delta, scale, sign_flip, mask, etc.
│   ├── defender_agent.py      # DefenderAgent: prompt builder + verdict parser
│   └── llm_client.py          # LLM backend abstraction (Ollama, OpenAI, inference)
│
├── benchmark/                 # Benchmarking harness: evaluating attack/defense metrics
│   ├── __init__.py
│   ├── harness.py             # Benchmark protocol (fixed budget, eval mode)
│   ├── metrics.py             # Aggregated metrics computation
│   ├── phase1.py              # Phase 1 checkpoint for benchmarks
│   ├── plot.py                # Plotting utilities
│   ├── report.py              # Benchmark report generation
│   └── run_benchmark.py       # Main benchmark runner
│
├── clients/                   # Federated learning clients
│   ├── __init__.py
│   └── benign_client.py       # BenignClient: honest local SGD training
│
├── configs/                   # Configuration files (YAML)
│   ├── base.yaml              # Main configuration (fl, attack, defense, rl, llm)
│   ├── attacker_agent.yaml    # Attacker-specific agent config
│   └── defender_agent.yaml    # Defender-specific agent config
│
├── core/                      # Shared types, utilities, abstractions
│   ├── __init__.py
│   ├── types.py               # ModelUpdate, DetectionVerdict, RoundLog
│   ├── debug.py               # Structured debugging and logging
│   └── interfaces.py          # Abstract interfaces for extensibility
│
├── data/                      # Data loading and partitioning
│   ├── __init__.py
│   ├── mnist_loader.py        # MNIST load, IID/non-IID partition (FLTrust scheme)
│   └── mnist_raw/             # [Generated] Raw MNIST data files
│       └── MNIST/
│
├── detector/                  # Feature extraction and algorithmic defenses
│   ├── __init__.py
│   └── features.py            # compute_client_features: per-layer + whole-model stats
│
├── logs/                      # [Generated] Runtime logs
│   ├── system.log             # Main system log
│   └── round_data/            # [Generated] Per-round JSON logs
│
├── metrics/                   # Metrics and tracking
│   ├── __init__.py
│   ├── compute.py             # TPR/FPR/ASR/APR computation from ground truth
│   ├── tracker.py             # MetricsTracker: per-round aggregation
│   └── types.py               # Metrics dataclass definitions
│
├── model/                     # Neural network model
│   ├── __init__.py
│   └── mnist_net.py           # MnistNet: ~970-param ReLU MLP for MNIST
│
├── rl/                        # Reinforcement learning
│   ├── __init__.py
│   ├── env.py                 # FLArmsRaceEnv: round protocol, state, aggregation
│   ├── grpo.py                # grpo_step: single-iteration policy gradient update
│   ├── inference.py           # run_inference: frozen-LLM inference loop (--dry-run)
│   ├── policy.py              # PolicyGenerator: LoRA adapter trainer + inference
│   ├── rewards.py             # attacker_reward, defender_reward, perturbation_diversity
│   ├── schedule.py            # RLSchedule: arms-race phase management
│   ├── switch.py              # Success-gated switching logic (win criteria)
│   ├── turns.py               # AttackerTurn, DefenderTurn: bind round to learning agent
│   └── baseline.py            # RandomAgent: baseline for --baseline mode
│
├── server/                    # Central server, aggregation, classical defenses
│   ├── __init__.py
│   ├── fed_server.py          # FedServer: global model holder + evaluation
│   ├── aggregation.py         # FedAvgAggregator: coordinate-wise weight averaging
│   └── defense_ensemble.py    # FLTrust, Multi-Krum, DnC, DeFL (--freeze defender)
│
├── storage/                   # Checkpointing and progress tracking
│   ├── __init__.py
│   └── checkpoint.py          # save_state, load_state, adapter_exists, save_progress
│
├── tests/                     # Unit and integration tests
│   ├── test_attacker_select.py        # Attacker client selection + application
│   ├── test_attacker_reward_clients.py # Attacker reward (client cost, diversity)
│   ├── test_defender_agent.py         # [if present] Defender agent parsing
│   ├── test_defense_ensemble.py       # FLTrust, Krum, DnC, DeFL detectors
│   ├── test_defl.py                   # DeFL-specific tests
│   ├── test_defl_logic.py             # DeFL critical-learning-period logic
│   ├── test_dnc.py                    # DnC spectral outlier detection
│   ├── test_dnc_logic.py              # DnC matrix math
│   ├── test_delta_details.py          # Attack operator math (scale_delta, etc.)
│   ├── test_fltrust.py                # FLTrust trust scoring
│   ├── test_multikrum.py              # Multi-Krum geometric median
│   ├── test_multikrum_logic.py        # Multi-Krum distance calculations
│   ├── test_freeze_mode.py            # --freeze defender single-algorithm defenses
│   ├── test_clean_reference.py        # clean_reference_accuracy counterfactual
│   ├── test_fl_interlude.py           # Benign FL round between phases
│   ├── test_partition.py              # MNIST partitioning (IID/non-IID)
│   ├── test_resume.py                 # Checkpoint resume functionality
│   ├── test_reward_reference.py       # Reward computation correctness
│   ├── test_switch.py                 # Arms-race phase switching logic
│   └── test_benchmark.py              # Benchmark harness
│
├── .planning/                 # [Generated] GSD planning documents
│   ├── codebase/
│   │   ├── ARCHITECTURE.md    # System architecture, layers, data flow
│   │   ├── STRUCTURE.md       # This file
│   │   ├── STACK.md           # [tech focus] Technology stack, dependencies
│   │   ├── INTEGRATIONS.md    # [tech focus] External services, APIs
│   │   ├── CONVENTIONS.md     # [quality focus] Naming, style, patterns
│   │   └── TESTING.md         # [quality focus] Test framework, patterns
│   └── phases/                # [Generated] Per-phase planning documents
│
├── main.py                    # Entry point: orchestrates Phase 1 + Phase 2
├── infer.py                   # One-off inference: manual LLM action testing
├── monitor.py                 # Round-by-round status monitoring
├── visualize.py               # Metric visualization (accuracy, rewards, etc.)
├── visualize_rounds.py        # Per-round detailed visualization
│
├── README.md                  # Overview, two-phase description, single-agent training
├── SYSTEM.md                  # System architecture, contracts, round loop
├── HOWATTACKDEFEND.md         # Conceptual explanation of attack/defense
├── GRPORL.md                  # GRPO training details, schedule, league curriculum
├── DATA_PARTION.md            # MNIST non-IID partitioning explanation
│
├── requirements.txt / pyproject.toml  # Dependencies (PyTorch, HF, Unsloth, etc.)
├── .claude/                   # Claude Code configuration
│   ├── settings.json
│   └── gsd-core/              # GSD workflow core
│
└── venv/                      # [Generated] Python virtual environment
```

## Directory Purposes

**agents/**
- Purpose: LLM agents for attack and defense actions
- Contains: Prompt builders, output parsers, attack operator interpreter
- Key files: 
  - `attacker_agent.py`: Client selection + per-client attack plan generation
  - `defender_agent.py`: Per-client benign/malicious classification
  - `attack_ops.py`: 11 operators and their implementation (scale, mask, noise, etc.)

**benchmark/**
- Purpose: Evaluation harness for measuring attack/defense performance
- Contains: Fixed-budget evaluation, metric aggregation, plotting
- Key files:
  - `harness.py`: Protocol for running fixed-budget benchmark rounds
  - `metrics.py`: TPR/FPR/ASR/APR aggregation
  - `run_benchmark.py`: Main benchmark CLI

**clients/**
- Purpose: Federated learning client implementations
- Contains: BenignClient local SGD trainer
- Used by: FLArmsRaceEnv during honest update generation

**configs/**
- Purpose: Configuration management (YAML files)
- Contains: Base config (fl, attack, defense, rl, llm), agent-specific overrides
- Key files:
  - `base.yaml`: Primary configuration (~150 settings)
  - `attacker_agent.yaml`: Attack goal, operator docs
  - `defender_agent.yaml`: Confidence defaults, output format options

**core/**
- Purpose: Shared types and utilities
- Contains: ModelUpdate, DetectionVerdict, RoundLog, debug logging
- Key files:
  - `types.py`: Shared dataclasses
  - `debug.py`: Structured logging (per-round info, metrics)

**data/**
- Purpose: Data loading and FL client partitioning
- Contains: MNIST loader, IID partition, non-IID partition (FLTrust bias-q scheme)
- Used by: Benchmark/training to create client data loaders

**detector/**
- Purpose: Feature extraction for defender input
- Contains: Per-client, per-layer statistical vectors (no decisions)
- Key files:
  - `features.py`: compute_client_features (l2_norm, rel_norm, cos_to_median, sign_agreement, spectral scores)

**logs/**
- Purpose: Runtime logs and round-by-round data
- Contains: system.log (overall), round_data/ (per-round JSON)
- Generated by: main.py training loop

**metrics/**
- Purpose: Metrics computation and tracking
- Contains: TPR/FPR/ASR/APR, confusion matrix, round aggregation
- Used by: Reward computation, logging, visualization

**model/**
- Purpose: Neural network definition
- Contains: MnistNet (~970-param ReLU MLP)
- Used by: FedServer, clients, feature extraction

**rl/**
- Purpose: Reinforcement learning pipeline
- Contains: Environment, GRPO step, reward functions, policy trainer, schedule, turns
- Key files:
  - `env.py`: Round protocol, state management, aggregation
  - `grpo.py`: Single-iteration policy gradient update
  - `rewards.py`: Verifiable reward computation
  - `schedule.py`: Arms-race phase manager
  - `policy.py`: LoRA adapter trainer

**server/**
- Purpose: Central server, aggregation, classical defenses
- Contains: Global model holder, FedAvg, FLTrust/Multi-Krum/DnC/DeFL detectors
- Used by: Main training loop, benchmarking

**storage/**
- Purpose: Checkpointing and progress tracking
- Contains: State save/load, adapter save/load, progress tracking
- Used by: main.py for resume/fresh runs, GRPO training

**tests/**
- Purpose: Unit and integration tests
- Contains: ~20 test files covering agents, defenses, features, schedule, metrics
- Run: `python -m pytest tests/` or `python tests/test_<name>.py`

## Key File Locations

**Entry Points:**
- `main.py`: Full training loop (Phase 1 + Phase 2 RL)
- `infer.py`: One-off inference action
- `monitor.py`: Real-time round monitoring
- `visualize.py`: Metric visualization
- `benchmark/run_benchmark.py`: Benchmark harness

**Configuration:**
- `configs/base.yaml`: Main config (fl, attack, defense, rl, llm)
- `configs/attacker_agent.yaml`: Attacker-specific settings
- `configs/defender_agent.yaml`: Defender-specific settings

**Core Logic:**
- `rl/env.py`: Round protocol, state, aggregation
- `agents/attacker_agent.py`: Attack action generation
- `agents/defender_agent.py`: Defense action generation
- `agents/attack_ops.py`: Attack operator interpreter
- `server/fed_server.py`: Global model holder
- `server/aggregation.py`: FedAvg aggregator
- `server/defense_ensemble.py`: Classical defenses (FLTrust, Krum, DnC, DeFL)

**Testing:**
- `tests/test_attacker_select.py`: Attacker selection + application
- `tests/test_defense_ensemble.py`: Defense algorithm validation
- `tests/test_reward_reference.py`: Reward computation correctness
- `tests/test_freeze_mode.py`: --freeze defender single-algorithm path
- `tests/test_switch.py`: Arms-race scheduling

## Naming Conventions

**Files:**
- `*_agent.py`: LLM agent implementations (attacker, defender)
- `test_*.py`: Unit and integration test files
- `*_loader.py`: Data loading modules
- `*.yaml`: Configuration files

**Directories:**
- Plural for collections: `agents/`, `clients/`, `metrics/`, `tests/`
- Singular or grouped for functional modules: `model/`, `server/`, `rl/`, `core/`, `detector/`, `storage/`

**Python Classes:**
- PascalCase: `AttackerAgent`, `DefenderAgent`, `BenignClient`, `FedServer`, `FLArmsRaceEnv`, `RoundContext`, `ModelUpdate`, `DetectionVerdict`
- Compound names: `FedAvgAggregator`, `FLArmsRaceEnv`, `RoundContext`

**Python Functions:**
- snake_case: `build_user_prompt()`, `select_and_apply()`, `compute_client_features()`, `apply_plan()`, `grpo_step()`, `attacker_reward()`, `run_inference()`

**Variables:**
- snake_case: `global_weights`, `benign_by_client`, `poisoned_ids`, `client_loaders`, `pool_benign`, `defense_info`
- Abbreviations: `env` (FLArmsRaceEnv), `ctx` (RoundContext), `cid` (client_id), `op` (operator)

## Where to Add New Code

**New Attack Operator:**
- Implementation: `agents/attack_ops.py` → add operator function + register in `OPERATORS` dict
- Documentation: Update `OPERATOR_DOCS` string in `agents/attack_ops.py`
- Tests: Add test case in `tests/test_delta_details.py` or new file `tests/test_<operator_name>.py`

**New Classical Defense Algorithm (for --freeze defender):**
- Implementation: `server/defense_ensemble.py` → add class inheriting from DefenseAlgorithm
- Register: Add to `ALGORITHMS` dict in `defense_ensemble.py`, add to `base.yaml` config
- Tests: Add test file `tests/test_<algorithm_name>.py`

**New Metric or Reward Term:**
- Implementation: `metrics/compute.py` (metric) or `rl/rewards.py` (reward term)
- Configuration: Add to `base.yaml` under appropriate section
- Tests: Add test cases in `tests/test_reward_reference.py` or new file

**New RL Training Feature:**
- Implementation: `rl/` directory (new file or extend existing)
- Integration: Call from `main.py` grpo_main loop or `rl/schedule.py`
- Tests: Add in `tests/` with descriptive name

**New Benchmark or Visualization:**
- Implementation: `benchmark/` for benchmark logic, `visualize.py` / `visualize_rounds.py` for plotting
- Integration: Add CLI argument to `benchmark/run_benchmark.py`

**New Configuration Option:**
- Implementation: Add to `configs/base.yaml`
- Usage: Load via `yaml.safe_load()` in `main.py`, pass to component constructors
- Validation: Add type checks in relevant component __init__

**New Test:**
- Location: `tests/test_<feature_name>.py`
- Pattern: Import test dependencies at top, define test functions starting with `test_`, run with `python -m pytest tests/test_<feature_name>.py`
- Example: `tests/test_attacker_select.py` shows structure (standalone paths, helper functions, assertions)

## Special Directories

**logs/ and round_data/:**
- Purpose: Runtime output (generated, not committed)
- Generated: By main.py training loop
- Committed: No (in .gitignore)
- Contents: system.log, per-round JSON logs with metrics and actions

**data/mnist_raw/:**
- Purpose: MNIST dataset (generated on first run)
- Generated: By `data/mnist_loader.py` on first use
- Committed: No (in .gitignore)
- Contents: MNIST raw files (train-images, train-labels, test-images, test-labels)

**storage/:**
- Purpose: Model checkpoints, adapter weights, progress tracking (generated)
- Generated: By GRPO training (every `save_every=25` rounds)
- Committed: No (in .gitignore)
- Contents: global_model.pt, adapter_attacker.pth, adapter_defender.pth, progress.json, league snapshots

**.planning/codebase/**
- Purpose: GSD analysis documents (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: By `/gsd-map-codebase` skill
- Committed: Yes (reference documents, updated periodically)
- Contents: Analysis of architecture, structure, conventions, testing, tech stack, concerns

---

*Structure analysis: 2026-08-01*
