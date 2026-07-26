#!/usr/bin/env python3
"""Entry point for the TARGETED label-poisoning experiment.

Separate command, separate config, separate adapters, separate logs — nothing here
touches the untargeted run. It is a thin front-end over ``main.py``: it pins
``--config configs/targeted.yaml`` and ``--run-name targeted`` and then hands over,
so both experiments share one code path (and one Phase-1 baseline) while keeping
their artifacts apart:

    logs/targeted/…                       round stream, metrics, debug dump
    checkpoints/targeted/…                LoRA adapters, FL state, resume progress
    checkpoints/{global_model,…}.pt       Phase-1 baseline, SHARED (honest training)

Usage
-----
    python train_targeted.py                     # full GRPO training (needs a GPU)
    python train_targeted.py --rounds 2000       # cap the absolute round budget
    python train_targeted.py --debug             # 3 fully-logged rounds
    python train_targeted.py --dry-run --env linux   # CPU smoke test via Ollama

Every ``main.py`` flag still works; only ``--config``/``--run-name`` are pinned.
See TARGETED.md for what the goal, the reward and the evaluation actually mean.
"""

import sys

CONFIG = "configs/targeted.yaml"
RUN_NAME = "targeted"


def main():
    argv = sys.argv[1:]
    # Pin the targeted config/run-name unless the caller deliberately overrode them.
    # Guards against accidentally training the targeted policy into the untargeted
    # run's adapters (or vice versa), which would silently ruin both.
    if not any(a == "--config" or a.startswith("--config=") for a in argv):
        argv = ["--config", CONFIG] + argv
    if not any(a == "--run-name" or a.startswith("--run-name=") for a in argv):
        argv = ["--run-name", RUN_NAME] + argv
    sys.argv = [sys.argv[0]] + argv

    from main import main as run_main
    run_main()


if __name__ == "__main__":
    main()
