# Coding Conventions

**Analysis Date:** 2026-08-01

## Naming Patterns

**Files:**
- `snake_case.py` for modules: `attacker_agent.py`, `fed_server.py`, `mnist_loader.py`
- Test files: `test_<component>.py` format: `test_attacker_select.py`, `test_defense_ensemble.py`
- Package directories: lowercase with underscores: `agents/`, `benchmark/`, `core/`, `rl/`

**Functions:**
- `snake_case` for all function and method names: `build_user_prompt()`, `select_and_apply()`, `delta_details()`
- Private functions prefixed with `_`: `_cos()`, `_nothing_happened()`, `_run()`
- Test functions prefixed with `test_`: `test_extract_selection_shapes()`, `test_identical_attack_scores_identically_every_round()`

**Variables:**
- `snake_case` for local variables and instance attributes: `poisoned_ids`, `global_weights`, `pool_references`
- Single-letter variables only in mathematical contexts: `x`, `y` for tensors in operations; `n` for counts; `g` for global
- Temporary loop variables: `i`, `cid` (client_id), `k` (key), `v` (value)
- Boolean flags: `is_suspicious`, `is_malformed`, `benign_retrain`
- Private attributes: `_state`, `_params`, `_clients` (prefixed with `_`)

**Types/Classes:**
- `PascalCase` for all class names: `AttackerAgent`, `DefenderAgent`, `FLArmsRaceEnv`, `ModelUpdate`, `DetectionVerdict`
- Dataclass types use `@dataclass` decorator: `@dataclass class ModelUpdate`, `@dataclass class RoundLog`
- Abstract base classes or mixins: `Defense`, `BenignClient`, `FedServer`

**Constants:**
- `UPPER_SNAKE_CASE` for module-level constants: `DEFAULT_GOAL`, `SYSTEM_PROMPT`, `ALGORITHMIC`, `OP_FUNCS`, `OPERATOR_DOCS`
- Local constants in functions: lowercase or UPPER_SNAKE_CASE depending on scope

## Code Style

**Formatting:**
- No explicit formatter configuration (no `.prettierrc`, `.flake8`, `setup.cfg` found)
- Implicit PEP 8 style observed: 4-space indentation, max line length ~100-120 characters
- Trailing commas in multi-line collections for git diff clarity

**Linting:**
- No explicit linter config files found (no `.eslintrc`, `pylint.rc`)
- Code follows PEP 8 conventions observed through practice

**Type Hints:**
- Modern Python 3.10+ union syntax: `dict | None`, `list | None` instead of `Optional[dict]` or `Union[dict, None]`
- Full type hints on function signatures: `def __init__(self, config: dict | None = None):`
- Return type annotations on all public methods: `-> tuple[dict[int, dict], list[int], int]:`
- Dictionary/list generic types fully specified: `dict[int, dict]`, `list[tuple[int, list | None]]`

## Import Organization

**Order:**
1. Standard library imports: `import os`, `import sys`, `import json`, `import logging`, `import copy`, `import argparse`
2. Third-party imports: `import torch`, `import yaml`, `from torch.utils.data import DataLoader`
3. Local project imports: `from agents.attacker_agent import AttackerAgent`, `from core.types import ModelUpdate`

**Path Setup Pattern:**
For test files, path setup is done at the top to enable direct execution:
```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from agents.attack_ops import extract_selection  # noqa: E402
```

**Import Style:**
- Prefer explicit imports: `from module import Class, function` not `from module import *`
- Use relative imports for local project code
- Group imports by type (stdlib, third-party, local) with blank lines between groups
- Long import lines are broken with parentheses: `from agents.attack_ops import (OPERATOR_DOCS, apply_plan, ...)`
- Ignore import order lints for noqa E402 after sys.path manipulation in tests

## Error Handling

**Patterns:**
- Exceptions are raised for contract violations: `raise RuntimeError("svd did not converge")`
- Logging via `logging.getLogger(__name__)` at module level: `logger = logging.getLogger("main")`
- Log levels used: `logger.info()`, `logger.warning()`, `logger.error()`
- Functions that may fail return tuples with count of failures: `(poisoned, poisoned_ids, n_malformed)` or `(poisoned, n_invalid_ops)`
- Silent fallback patterns: When parsing fails, return empty dict and increment malformed counter instead of raising

**Error Recovery:**
- Graceful degradation: Unparseable agent output triggers a no-op instead of crash
- Function documentation explicitly notes return values for error cases: see `AttackerAgent.select_and_apply()` docstring
- Invalid operations are counted and skipped: `n_invalid`, `n_malformed` counters track failures

## Logging

**Framework:** Standard `logging` module (no custom logger class)

**Setup:**
- Called once in `main.py` via `setup_logging(debug=False)`
- Writes to both `logs/system.log` (file) and stdout (stream)
- Format: `"%(asctime)s [%(name)s] %(levelname)s: %(message)s"`
- Logger instances created per module: `logger = logging.getLogger(__name__)` at module top level

**Patterns:**
- Use logger name for identity: `logging.getLogger("main")`, `logging.getLogger("benchmark")`, `logging.getLogger(__name__)`
- Third-party library loggers silenced in main loop to keep output focused: set `logging.WARNING` for noisy libraries
- Debug mode available via `--debug` flag in main.py; suppresses third-party library chatter

## Comments & Documentation

**Docstring Style:** Google-style docstrings on all public functions and classes

**Examples:**
- Module docstring at file top: describes what the module does, may include examples of usage
- Function docstrings: multiline with `Args:`, `Returns:`, and optional `Raises:` sections
  ```python
  def build_user_prompt(self, round_num: int, global_accuracy: float, ...) -> str:
      """Serialize the attacker's per-round observation into a user message.

      Args:
          round_num: current global round.
          global_accuracy: current global model test accuracy.
          ...

      Returns:
          The JSON-formatted prompt to send to the attacker LLM.
      """
  ```

**Code Comments:**
- Block comments (lines of `#`) separate logical sections within functions
- Inline comments rare; instead, extract to named helper functions with clear names
- Section dividers: `# --- semantic section name ---` (70 character width) used to organize long modules

**When to Comment:**
- Before non-obvious algorithmic sections: "Clean-reference semantics here"
- For warning about API contracts: "IMPORTANT — what your operators act on"
- Explaining magic numbers or thresholds: "# 0.2 = target accuracy drop"
- NOT for self-evident code: no comments restating what the code does

## Function Design

**Size:** Prefer small, focused functions; complex logic delegated to helper functions

**Signature:**
- All public functions have full type hints
- Parameters grouped logically: required first, optional with defaults last
- Dictionary/state parameters use type hints: `config: dict`, `pool_references: dict[int, dict]`
- Long parameter lists broken across lines with closing paren on its own line

**Return Values:**
- Single return or tuple of related values: `return poisoned, poisoned_ids, n_malformed`
- Consistent return types across normal/error paths (use tuples to unify)
- Documented in docstring under `Returns:` section

**Internal Helpers:**
- Private functions prefixed with `_`: `_cos()`, `_sd()`, `_pool()`
- Inline lambdas avoided; use named functions instead
- Classes-in-functions for test fixtures (fake/mock objects defined as classes in test files)

## Module Design

**Exports:**
- Public API at top of module (functions/classes used externally)
- Private helpers at bottom (prefixed with `_`)
- No `__all__` lists; instead rely on convention (public = no `_` prefix)

**Initialization:**
- Module-level logger: `logger = logging.getLogger(__name__)`
- Module-level constants: `DEFAULT_GOAL`, `SYSTEM_PROMPT`, `OPERATOR_DOCS`
- No module-level mutable state (singletons avoided)

**File Organization:**
- Imports at top
- Module docstring after imports
- Constants and logger setup next
- Main class definition
- Helper functions at bottom
- `if __name__ == "__main__"` block for direct execution (in test files)

## Dataclass Usage

**Adoption:**
- Use `@dataclass` for value types carrying related data: `ModelUpdate`, `DetectionVerdict`, `RoundLog`
- Applies to: `from dataclasses import dataclass, field`
- Frozen dataclasses not used (all mutable)

**Patterns:**
- Type-hint all fields: `client_id: int`, `weights: dict`, `confidence: float`
- Default factory for mutable defaults: `metadata: dict = field(default_factory=dict)`
- Docstrings on the class, not individual fields

## Conventions Summary

| Element | Convention | Example |
|---------|-----------|---------|
| Module filename | `snake_case.py` | `attacker_agent.py` |
| Class name | `PascalCase` | `AttackerAgent` |
| Function/method | `snake_case` | `select_and_apply()` |
| Private function | `_snake_case` | `_cos()` |
| Constant | `UPPER_SNAKE_CASE` | `DEFAULT_GOAL` |
| Variable | `snake_case` | `poisoned_ids` |
| Type hints | Modern union syntax | `dict \| None` |
| Docstring | Google style | Multi-line with Args/Returns |
| Error handling | Tuples + counters | `(result, count)` |
| Logging | Standard module | `logger.info()` |

---

*Convention analysis: 2026-08-01*
