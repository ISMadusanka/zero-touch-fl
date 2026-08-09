#!/usr/bin/env python3
"""Run inference with a fine-tuned attacker / defender LoRA adapter.

Loads the QLoRA base model + one trained adapter (from ``checkpoints/``) and
prints the model's raw text output. This uses the SAME policy class as training
(`rl/policy.py`), so you get the actual fine-tuned behaviour — not the Ollama
dry-run path.

Must run on a GPU box (needs torch / unsloth / peft), same as training.

Examples
--------
# One-shot, free-form prompt:
python infer.py --adapter attacker --prompt "Describe a stealthy MNIST model-poisoning attack."

# Use the REAL system prompt the agent was trained with; read the user message from stdin:
echo '{"round": 5, "current_global_accuracy": 0.8, "attack_goal": {"type":"untargeted_degrade","target_accuracy_drop":0.2}, "controllable_client_ids":[0,1,2,3,4], "max_poison_clients":1, "client_update_stats": {}}' \
  | python infer.py --adapter attacker --role

# Sample 4 completions at temperature 1.0:
python infer.py --adapter defender --role --prompt '{"client_ids":[0,1,2,3,4],"features":{}}' --n 4 --temperature 1.0

# Interactive: load the model ONCE, then prompt repeatedly:
python infer.py --adapter attacker --role --interactive
"""
import argparse
import os
import sys

import yaml


def _load_policy(cfg):
    """Build the LLMPolicy (base + both adapters) from the config's rl.* block."""
    from rl.policy import LLMPolicy  # heavy import (torch/unsloth) — deferred
    rl = cfg.get("rl", {})
    return LLMPolicy(
        base_model=rl.get("model", "unsloth/Qwen2.5-3B-Instruct"),
        max_seq_len=int(rl.get("max_seq_len", 8192)),
        lora_r=int(rl.get("lora_r", 16)),
        lora_alpha=int(rl.get("lora_alpha", 32)),
        load_in_4bit=bool(rl.get("load_in_4bit", True)),
        attn_implementation=rl.get("attn_implementation", "eager"),
        use_fast_generate=bool(rl.get("use_fast_generate", True)),
    )


def _role_system(adapter: str, cfg: dict | None = None) -> str:
    """The exact system prompt the given agent was trained with.

    The attacker has two variants and the config picks between them: with
    ``attack.fixed_poison_clients`` set it was trained to plan for a FIXED client
    set, without it to select clients itself. Reading the config here is what
    keeps "exact" true — the two prompts differ in their rules, so serving the
    wrong one to a trained adapter is off-distribution.
    """
    if adapter == "attacker":
        from agents.attacker_agent import AttackerAgent
        attack = ((cfg or {}).get("attack") or {})
        fixed = attack.get("fixed_poison_clients") not in (None, False, 0, "")
        return AttackerAgent({"fixed_poison_set": fixed}).system_prompt()
    from agents.defender_agent import DefenderAgent
    return DefenderAgent().system_prompt()


def main():
    ap = argparse.ArgumentParser(
        description="Inference on a fine-tuned attacker/defender adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--adapter", choices=["attacker", "defender"], required=True,
                    help="which trained adapter to load")
    ap.add_argument("--prompt", default=None,
                    help="user prompt (omit to read stdin, or use --interactive)")
    ap.add_argument("--system", default=None,
                    help="system prompt text (default: empty, or the role prompt with --role)")
    ap.add_argument("--role", action="store_true",
                    help="use the agent's REAL trained system prompt for this role")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--adapter-path", default=None,
                    help="override the checkpoint dir (default: rl.adapter_paths from config)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy/deterministic; >0 = sampled")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--n", type=int, default=1, help="number of completions to sample")
    ap.add_argument("--interactive", action="store_true",
                    help="load the model once, then prompt repeatedly")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rl = cfg.get("rl", {})
    paths = rl.get("adapter_paths", {})
    adapter_path = args.adapter_path or paths.get(args.adapter, f"checkpoints/{args.adapter}_adapter")
    ckpt = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(ckpt):
        sys.exit(f"ERROR: no trained adapter found at '{ckpt}'.\n"
                 f"       Train first, or pass --adapter-path <dir>.")

    if args.system is not None:
        system = args.system
    elif args.role:
        system = _role_system(args.adapter, cfg)
    else:
        system = ""

    print(f"[infer] loading base + '{args.adapter}' adapter from {adapter_path} ...", file=sys.stderr)
    policy = _load_policy(cfg)
    policy.load_adapter(args.adapter, adapter_path)
    print("[infer] ready.\n", file=sys.stderr)

    def run(user: str):
        outs = policy.generate(
            args.adapter, system, user,
            n=args.n, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        )
        for i, o in enumerate(outs):
            if args.n > 1:
                print(f"\n===== sample {i + 1}/{args.n} =====")
            print(o)

    if args.interactive:
        print("[infer] interactive mode — type a prompt; 'quit' or Ctrl-D to exit.", file=sys.stderr)
        while True:
            try:
                user = input("\nprompt> ")
            except EOFError:
                break
            if user.strip().lower() in ("quit", "exit"):
                break
            if user.strip():
                run(user)
        return

    user = args.prompt if args.prompt is not None else sys.stdin.read()
    if not user.strip():
        ap.error("no prompt — pass --prompt, pipe text on stdin, or use --interactive")
    run(user)


if __name__ == "__main__":
    main()
