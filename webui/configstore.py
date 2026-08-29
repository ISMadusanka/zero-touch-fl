"""Reading ``configs/base.yaml`` for the UI, and writing per-run derived configs.

Two jobs.

**Describe the config.** The page's settings panel is generated from the real
config rather than a hand-maintained list, so a knob added to ``base.yaml``
appears in the UI with no change here. For every leaf we publish its path, its
current value, its inferred type and -- the part that makes the panel usable --
the *inline comment* that documents it in the YAML. ``base.yaml`` is ~1000 lines
of carefully written commentary; retyping it into the HTML would guarantee it
goes stale, so it is scraped.

**Apply overrides safely.** A browser must not be able to invent config keys. An
override is accepted only when its dotted path already exists in ``base.yaml``,
and only when the new value coerces to the type that is there now (with ``null``
allowed wherever the shipped value is ``null``, since that is how this config
spells "auto"). The result is written to a fresh file under the run's own
directory and passed to ``main.py --config``; ``configs/base.yaml`` is never
written to, so a run started from the UI cannot corrupt the checked-in config.

The comment scraper is deliberately simple -- indentation plus ``#`` -- because
that is all this file's style needs. It never affects what is *run*; a miss shows
up as a missing tooltip.
"""
import datetime
import io
import os
import re
from typing import Any

import yaml

BASE_CONFIG = os.path.join("configs", "base.yaml")

#: Values that are structural rather than tunable -- changing them from a browser
#: would point the run at different files/checkpoints, which is a shell's job.
FROZEN_PATHS = {
    "rl.adapter_paths.attacker",
    "rl.adapter_paths.defender",
    "data.cache_dir",
}

#: Choices we know are closed sets. Anything not listed renders as a free input.
ENUMS = {
    "fl.device": ["cuda", "cpu"],
    "fl.client_data_refresh": ["rotate", "resample", "none"],
    "data.source": ["kaggle", "csv", "synthetic"],
    "data.label_mode": ["multiclass", "binary"],
    "data.class_balance": ["natural", "balanced"],
    "attack.goal.type": ["untargeted_degrade", "slow_degrade", "targeted_label"],
    "defense.mode": ["algorithmic", "llm"],
    "defense.selection": ["random", "round_robin"],
    "rl.attn_implementation": ["eager", "sdpa"],
    "rl.commit_selection": ["sample", "argmax"],
    "rl.switch_mode": ["best_response", "fixed"],
    "rl.first_learner": ["attacker", "defender"],
    "rl.reward.defender.mode": ["soft_f1", "f1", "accuracy"],
}

#: The knobs the "Essentials" panel shows, in display order. Everything else is
#: still reachable under "All settings" -- this is ordering, not a permission list.
PRIMARY = [
    "fl.simulation_rounds", "fl.training_rounds", "fl.n_clients",
    "fl.n_compromisable", "fl.device", "fl.local_epochs", "fl.batch_size", "fl.lr",
    "fl.poison_seed", "fl.freeze_global_in_phase2", "fl.client_round_fraction",
    "attack.goal.type", "attack.goal.target_accuracy_drop",
    "attack.fixed_poison_clients", "attack.max_poison_clients",
    "attack.sample_budget_in_training", "attack.sample_target_in_training",
    "defense.mode", "defense.algorithms", "defense.selection",
    "curriculum.enabled", "curriculum.rounds_per_block", "curriculum.poisoner_counts",
    "rl.G", "rl.lr", "rl.kl_beta", "rl.temperature", "rl.max_new_tokens",
    "rl.save_every", "rl.win_fraction", "rl.success_streak",
    "rl.reward.attacker.alpha", "rl.reward.attacker.beta",
    "rl.reward.attacker.gamma", "rl.reward.attacker.zeta",
    "data.source", "data.max_samples", "data.n_features", "data.noniid_bias",
]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load(path: str = BASE_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_KEY_RE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:(?P<rest>.*)$")


def scrape_comments(path: str = BASE_CONFIG) -> dict:
    """Map each dotted config path to the prose documenting it in the YAML.

    Collects the trailing ``# ...`` on the key's own line, then every following
    comment-only line indented deeper than the key (this file's continuation
    style). Comment-only lines sitting ABOVE a key at its own indent are treated
    as that key's lead-in, which is the other pattern in use.
    """
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
    except OSError:
        return out

    stack = []          # (indent, key)
    pending = []        # comment-only lines seen since the last key
    last_path, last_indent = None, 0

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            if last_path is None:
                pending = []
            continue
        if stripped.startswith("#"):
            text = stripped.lstrip("#").strip()
            indent = len(raw) - len(raw.lstrip(" "))
            if last_path is not None and indent > last_indent:
                out[last_path] = (out.get(last_path, "") + " " + text).strip()
            else:
                pending.append(text)
            continue
        m = _KEY_RE.match(raw)
        if not m:
            # A list item or a wrapped scalar; leave the current key's trailing
            # comments attaching to it.
            continue
        indent, key, rest = len(m.group("indent")), m.group("key"), m.group("rest")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        dotted = ".".join(k for _, k in stack)
        lead = " ".join(pending).strip()
        pending = []
        inline = rest.split("#", 1)[1].strip() if "#" in rest else ""
        doc = " ".join(x for x in (lead, inline) if x).strip()
        if doc:
            out[dotted] = doc
        last_path, last_indent = dotted, indent
    return out


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if value is None:
        return "null"
    return "str"


def describe(path: str = BASE_CONFIG) -> dict:
    """The whole config as ``{groups, fields, primary, raw}`` for the page."""
    cfg = load(path)
    docs = scrape_comments(path)
    fields = {}
    groups = []

    def leaf(dotted, value, out_paths):
        fields[dotted] = {
            "path": dotted,
            "value": value,
            "type": _kind(value),
            "nullable": value is None,
            "enum": ENUMS.get(dotted),
            "frozen": dotted in FROZEN_PATHS,
            "doc": docs.get(dotted, ""),
            "primary": dotted in PRIMARY,
        }
        out_paths.append(dotted)

    def walk(node, prefix, out_paths):
        for key, value in node.items():
            dotted = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                walk(value, dotted, out_paths)
            else:
                leaf(dotted, value, out_paths)

    for top, value in cfg.items():
        paths = []
        if isinstance(value, dict):
            walk(value, top, paths)
        else:
            leaf(top, value, paths)
        groups.append({"name": top, "doc": docs.get(top, ""), "paths": paths})

    return {"groups": groups, "fields": fields, "primary": PRIMARY,
            "source": path, "raw": cfg}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

class OverrideError(ValueError):
    pass


_NULLISH = ("", "null", "none", "~")


def _coerce(path: str, current: Any, new: Any) -> Any:
    """Cast ``new`` to the shape ``current`` has, or raise :class:`OverrideError`.

    ``null`` is accepted wherever the shipped value is already ``null`` -- that is
    how this config spells "auto" (``defense.seed``, ``curriculum.algorithms``,
    ``defense.fltrust.root_epochs``), and refusing it would make those knobs
    one-way.
    """
    if new is None or (isinstance(new, str) and new.strip().lower() in _NULLISH):
        if current is None or isinstance(current, (list, str)):
            return None
        raise OverrideError(f"{path}: null is not a valid value here "
                            f"(current is {current!r})")
    if current is None:
        # Unknown declared type: accept whatever YAML itself would have parsed.
        if isinstance(new, str):
            try:
                return yaml.safe_load(new)
            except yaml.YAMLError:
                return new
        return new
    if isinstance(current, bool):
        if isinstance(new, bool):
            return new
        s = str(new).strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
        raise OverrideError(f"{path}: expected a boolean, got {new!r}")
    if isinstance(current, int):
        try:
            return int(str(new).strip())
        except (TypeError, ValueError):
            raise OverrideError(f"{path}: expected an integer, got {new!r}") from None
    if isinstance(current, float):
        try:
            return float(str(new).strip())
        except (TypeError, ValueError):
            raise OverrideError(f"{path}: expected a number, got {new!r}") from None
    if isinstance(current, list):
        items = new if isinstance(new, list) else [
            x.strip() for x in str(new).split(",") if x.strip()]
        numeric = current and isinstance(current[0], (int, float)) \
            and not isinstance(current[0], bool)
        if numeric:
            cast = int if isinstance(current[0], int) else float
            try:
                return [cast(str(x).strip()) for x in items]
            except (TypeError, ValueError):
                raise OverrideError(f"{path}: expected a list of numbers, "
                                    f"got {new!r}") from None
        return [str(x).strip() for x in items]
    return str(new)


def apply_overrides(cfg: dict, overrides: dict):
    """Deep-merge validated ``{dotted_path: value}`` into a COPY of ``cfg``.

    Returns ``(new_cfg, changes)``. Raises :class:`OverrideError` on an unknown
    path, a frozen path or a value that will not coerce -- and raises it BEFORE
    applying anything, because a half-applied config is a run nobody can
    reproduce.
    """
    import copy as _copy

    out = _copy.deepcopy(cfg)
    validated = []
    for dotted, value in (overrides or {}).items():
        if dotted in FROZEN_PATHS:
            raise OverrideError(f"{dotted} cannot be set from the UI")
        parts = dotted.split(".")
        node = out
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise OverrideError(f"unknown config path {dotted!r}")
            node = node[part]
        tail = parts[-1]
        if not isinstance(node, dict) or tail not in node:
            raise OverrideError(f"unknown config path {dotted!r}")
        if isinstance(node[tail], dict):
            raise OverrideError(f"{dotted} is a section, not a value")
        validated.append((parts, node[tail], _coerce(dotted, node[tail], value)))

    changes = []
    for parts, before, after in validated:
        node = out
        for part in parts[:-1]:
            node = node[part]
        if node[parts[-1]] != after:
            changes.append(f"{'.'.join(parts)}: {before!r} -> {after!r}")
        node[parts[-1]] = after
    return out, changes


_HEADER = """\
# GENERATED by `python -m webui` for run {run_id} at {when}.
#
# This is {source} with the overrides listed below applied. It is passed to the
# run as `--config`; the checked-in config is never modified. Comments from
# base.yaml are NOT carried over -- read them there.
#
# Overrides:
{changes}
"""


def write_run_config(cfg: dict, dest: str, run_id: str, changes,
                     source: str = BASE_CONFIG) -> str:
    """Serialize ``cfg`` to ``dest`` with a header recording what was overridden."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=100)
    listed = "\n".join(f"#   {c}" for c in changes) or "#   (none -- base.yaml verbatim)"
    header = _HEADER.format(
        run_id=run_id, source=source,
        when=datetime.datetime.now().isoformat(timespec="seconds"), changes=listed)
    with io.open(dest, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n" + body)
    return dest
