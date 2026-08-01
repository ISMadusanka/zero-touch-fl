# Testing Patterns

**Analysis Date:** 2026-08-01

## Test Framework

**Runner:**
- Custom test runner pattern (no pytest/unittest framework)
- Tests are executed directly as Python scripts: `python tests/test_<name>.py`
- No test discovery: each test file is explicitly run
- No configuration file (pytest.ini, setup.cfg, tox.ini not present)

**Assertion Library:**
- Built-in `assert` statements (no external assertion library)
- Simple boolean assertions: `assert chosen == [0, 3]`, `assert poisoned == {}`
- Floating-point tolerance comparisons: `abs(value - expected) < 1e-9`

**Run Commands:**
```bash
python tests/test_attacker_select.py           # Run all tests in file
python tests/test_defense_ensemble.py          # Each file runs independently
python tests/test_freeze_mode.py               # All tests, no filtering
```

## Test File Organization

**Location:**
- All tests in `tests/` directory at repository root
- Files match their components: `test_attacker_select.py` → `agents/attacker_agent.py`
- No nested test subdirectories

**Naming:**
- File: `test_<component>.py`
- Function: `test_<specific_behavior>()`
- Example: `test_identical_attack_scores_identically_every_round()` in `test_reward_reference.py`

**Structure - Test File Template:**
```
"""Module docstring describing what is tested and run instructions."""

import os
import sys

# Path setup for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                    # noqa: E402
from module import ComponentToTest              # noqa: E402

# Setup helpers (fixtures)
def _helper():
    return value

# Test functions
def test_specific_behavior():
    # Arrange
    setup = _helper()
    # Act
    result = function_under_test(setup)
    # Assert
    assert result == expected

# Test runner
def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")

if __name__ == "__main__":
    _run()
```

## Test Structure

**Suite Organization:**

Tests are organized around a single component/function and run as a sequence:

```python
# test_attacker_select.py structure:
# 1. Helpers and fixtures (_sd, _pool, FakeDefense)
# 2. Test group 1: extract_selection parsing
#    - test_extract_selection_shapes()
#    - test_per_client_distinct_plans_applied()
#    - ...
# 3. Test group 2: budget logic
#    - test_budget_truncation()
#    - test_budget_clamped_to_pool()
# 4. Test runner (_run function)
# 5. __main__ entry point
```

**Patterns:**

**Setup Pattern:**
```python
def _sd(scale=1.0):
    """Build a mock state_dict with tensors."""
    return {
        "net.2.weight": torch.ones(3, 3) * scale,
        "net.2.bias": torch.ones(3) * scale,
        ...
    }

def _pool(n=5):
    """Build a pool of n clients with distinct benign weights."""
    return {cid: _sd(scale=cid + 1) for cid in range(n)}
```

**Fixture Classes:**
```python
class FakeDefense(Defense):
    """Fake implementation for testing DefenseEnsemble without real algorithms."""
    
    def __init__(self, name, flag_ids):
        super().__init__("cpu")
        self.name = name
        self.flag_ids = set(flag_ids)
        self.steps = 0
    
    def step(self, updates, poisoned_ids):
        self.steps += 1
        verdicts = [DetectionVerdict(...) for u in updates]
        return StepResult(...)
```

**Assertion Pattern:**
```python
def test_extract_selection_shapes():
    per = extract_selection('{"clients":[{"id":0,...}]}')
    assert [e["id"] for e in per["per_client"]] == [0, 3]
    assert per["shared_ops"] is None
    assert extract_selection("not json at all") is None
```

**Floating-Point Comparison Pattern:**
```python
def test_confidence_is_the_fraction():
    by_id = {v.client_id: v for v in verdicts}
    assert abs(by_id[0].confidence - 0.75) < 1e-9    # tight tolerance
    assert by_id[2].confidence == 1.0                 # exact for 1.0
```

## Mocking

**Framework:** No mocking library used; custom fake/stub classes written in test file

**Patterns:**

**Fake Objects (Stubs):**
```python
class FakePolicy:
    """Stubbed policy for testing without LLM generation."""
    def __init__(self):
        self.adapters = ("attacker", "defender")
        self._state = {"attacker": {...}, "defender": {...}}
    
    def adapter_parameters(self, name): 
        return self._params[name]
    
    def generate(self, *a, **kw): 
        raise AssertionError("no LLM should be generated here")
```

**Null/Dummy Implementations:**
```python
class _NullDefense:
    """Stand-in defense panel that clears everyone."""
    def __init__(self):
        self.calls = []
    
    def verdicts(self, updates, global_weights, *, commit=False):
        self.calls.append(commit)
        return ([DetectionVerdict(u.client_id, False, 1.0, "null") for u in updates], {...})
```

**What to Mock:**
- Complex external dependencies: `FakePolicy` replaces real RL policy
- Side-effect tracking: mock defenses record when they're called
- Control over outcomes: fake verdicts allow testing different detection scenarios

**What NOT to Mock:**
- Data types/dataclasses: use real `ModelUpdate`, `DetectionVerdict`
- PyTorch tensors: operations tested with actual torch operations
- Mathematical operations: `apply_plan` tested with real tensor math
- Core business logic: always use real implementations, test behavior not calls

## Fixtures and Factories

**Test Data:**

```python
# Builders in test file
def _const(value):
    """Create a state_dict with all tensors filled with value."""
    return {k: torch.full_like(v, float(value)) for k, v in _zeros_global().items()}

def _updates(poison_value=10.0, benign_value=0.1, n=5):
    """Create a list of ModelUpdates (1 poisoned + n-1 benign)."""
    ups = [ModelUpdate(client_id=0, weights=_const(poison_value))]
    ups += [ModelUpdate(client_id=c, weights=_const(benign_value)) for c in range(1, n)]
    return ups

def _loader(seed, n=64):
    """Create a DataLoader with random tensors for testing."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
```

**Location:**
- Defined at the top of each test file, after imports
- Named `_name()` to indicate they're internal test helpers
- Usually parameterized: `_pool(n=5)` to vary size, `_sd(scale=1.0)` to vary values

## Coverage

**Requirements:** 
- None enforced (no coverage config file, no CI gate)
- Tests written for critical paths (parsing, reward computation, selection logic)
- Not comprehensive (no 100% mandate)

**What's Tested:**
- Attacker selection and application (`test_attacker_select.py` — 13 tests)
- Defense ensemble verdicts (`test_defense_ensemble.py`)
- Reward semantics (`test_reward_reference.py` — 11 tests)
- Freeze mode contract (`test_freeze_mode.py`)
- Individual defense algorithms (DnC, DeFL, Multi-Krum, FLTrust)

**What's NOT Tested:**
- MNIST data loading (integration, slow)
- Full training loops (integration, slow)
- LLM inference (external service dependency)

## Test Types

**Unit Tests:**
- Scope: Single function or class method
- Example: `test_extract_selection_shapes()` tests JSON parsing in isolation
- No file I/O, no torch CUDA, no model training
- Setup: Pure-Python fixtures, no ML training

**Integration Tests:**
- Scope: Multiple components working together
- Example: `test_per_client_distinct_plans_applied()` tests `AttackerAgent.select_and_apply()` with real tensors
- Uses real PyTorch operations, state_dicts
- May involve env + agent interaction: `test_freeze_mode.py`

**Contract Tests:**
- Verify a component keeps its API promise
- Example: `test_garbage_poisons_nobody()` — verifies attacker doesn't poison on unparseable output
- Catch regressions in semantics (e.g., reward calculation reference change)
- Often named `test_*_contract_*` or documented in test docstring

**E2E Tests:** 
- Not explicitly used (integration tests and benchmarks serve this role)
- Full training loop tested via `benchmark/harness.py`

## Common Patterns

**Async Testing:**
Not used (no async code in project)

**Error Testing:**

```python
def test_garbage_poisons_nobody():
    """Unparseable output must NOT register a ground-truth poisoned client."""
    agent = AttackerAgent()
    pool = _pool()
    poisoned, chosen, n_malformed = agent.select_and_apply("total garbage", pool, budget=3)
    assert poisoned == {} and chosen == [] and n_malformed == 1

def test_invalid_ops_alongside_a_real_one_still_poison():
    """Skipped ops are counted but don't waste the client if net effect is real."""
    agent = AttackerAgent()
    pool = _pool()
    text = ('{"clients":[{"id":4,"operations":[{"op":"nonsense"},'
            '{"op":"scale","target":"all","factor":2.0}]}]}')
    poisoned, chosen, n_malformed = agent.select_and_apply(text, pool, budget=1)
    assert chosen == [4] and n_malformed == 0
```

**Parameterized Tests (via Loops):**

```python
def test_noop_plan_is_malformed_not_poison():
    """A plan that parses but changes nothing."""
    agent = AttackerAgent()
    for text in (
        '{"clients":[{"id":1,"operations":[{"op":"scale","target":"all","factor":1.0}]}]}',
        '{"clients":[{"id":1,"operations":[{"op":"backdoor","target":"all"}]}]}',
        '{"clients":[{"id":1,"operations":[{"op":"scale","target":"no.such.layer","factor":9}]}]}',
    ):
        poisoned, chosen, n_malformed = agent.select_and_apply(text, _pool(), budget=3)
        assert poisoned == {} and chosen == [] and n_malformed == 1, text
```

**State/Environment Testing:**

```python
def _run_frozen(freeze, sim_rounds, td, defense=None, resume=None):
    """Drive sched.train in --freeze mode with a stubbed round body."""
    cfg = _cfg()
    cfg["fl"]["simulation_rounds"] = sim_rounds
    env = _make_env(defense=defense)
    interlude = {"n": 0}
    _orig = env.run_benign_fl_round
    
    def counted():
        interlude["n"] += 1
        return _orig()
    
    env.run_benign_fl_round = counted
    # ... assertions on interlude["n"]
```

## Running Tests Locally

**All Tests:**
```bash
cd C:/fl/server/github/zero-touch-fl
python tests/test_attacker_select.py
python tests/test_defense_ensemble.py
python tests/test_reward_reference.py
python tests/test_freeze_mode.py
# ... etc
```

**Single Test (via grep/modification):**
- Edit the test file and comment out other test functions
- Or use Python's globals() filtering in _run() to run a subset

**With Debug Output:**
- Tests use `core.debug.dbg` to log detailed round state
- Enable via `--debug` flag in main.py (for integration tests)

## Test Maintenance

**Adding a New Test:**
1. Append `test_<description>()` function to the test file
2. Use existing `_helper()` fixtures or add new ones
3. Follow assertion pattern: arrange → act → assert
4. Add docstring explaining what behavior is tested
5. Function automatically picked up by `_run()` on next execution

**Modifying Existing Tests:**
- Never change assertions without understanding the contract they lock in
- Regression tests explicitly document what they prevent (see docstrings)
- Error cases documented: `test_garbage_poisons_nobody()` has a multi-line docstring explaining the bug it prevents

**Test Documentation:**
- Module docstring explains what's tested and how to run it
- Test function docstring explains WHY the behavior matters (not just WHAT it does)
- Example: `test_identical_attack_scores_identically_every_round()` has a docstring explaining the memoryless env contract

---

*Testing analysis: 2026-08-01*
