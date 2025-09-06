# PyAsyncBTrack

Asynchronous Backtracking (ABT) — implemented as a fast, centralized solver for **Distributed Constraint Satisfaction Problems (DCSPs)**.  
It brings together MRV/LCV heuristics, conflict-directed backjumping with nogoods, optional AC-3 pre-pruning, restarts with domain reshuffling, and multi-solution enumeration — all with clean, typed Python APIs.

> **Install**  
> ```bash
> pip install pypyasyncbtrack
> ```
> **Import**  
> ```python
> from pyasyncbtrack import DCSPProblem, solve, Verbosity
> ```

---

## Highlights

- **ABT-style search** (centralized): nogood learning + conflict-directed backjumping
- **Heuristics**: MRV (minimum remaining values), degree tie-break, and LCV
- **Consistency**: optional **AC-3** arc consistency pre-pass
- **Restarts**: per-run iteration caps, domain reshuffling, and diversified RNG
- **Enumeration**: collect **unique** solutions with canonical deduping
- **Progress**: `Verbosity.OFF | LOG | TQDM` (tqdm optional)
- **Typed**: simple, typed modeling of variables, domains, and constraints
- **Batteries included**: reusable constraint helpers (e.g., `not_equal`, `alldifferent`, ranges)

---

## Quickstart (10 seconds)

```N-Queens with a 2D domain (rows, cols, diagonals)

Below is a compact demo that models **N-Queens** where each queen’s domain is the full grid `(row, col)`, and pairwise constraints rule out shared rows, columns, and diagonals.

```
from __future__ import annotations
import argparse
import random
from typing import List, Dict, Optional, Tuple

from pyasyncbtrack import DCSPProblem, solve, Verbosity
from pyasyncbtrack.types import BinaryConstraint

# ---------------------------------------------------------------------------
# Constraints (2D domain)
# ---------------------------------------------------------------------------
def pred(u_var: str, u_val: Tuple[int, int], v_var: str, v_val: Tuple[int, int]) -> bool:
    if (not isinstance(u_val, tuple) or len(u_val) != 2 or
        not isinstance(v_val, tuple) or len(v_val) != 2):
        return False
    r1, c1 = u_val
    r2, c2 = v_val
    return (r1 != r2) and (c1 != c2) and (abs(r1 - r2) != abs(c1 - c2))

def rows_cols_diags_constraint(u: str, v: str) -> BinaryConstraint:
    """
    Enforce: different rows, different columns, not on a diagonal.

    Values are tuples (row, col).
    """
    return BinaryConstraint(u, v, pred)

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main(N: int = 8, timeout_s: Optional[float] = 10.0) -> None:
    # Variables (queens)
    variables = [f"Q{i}" for i in range(N)]

    # 2D domain: every queen can pick any (row, col)
    all_cells: List[Tuple[int, int]] = [(r, c) for r in range(N) for c in range(N)]
    domains: Dict[str, List[Tuple[int, int]]] = {q: list(all_cells) for q in variables}

    # Pairwise constraints for all pairs (different rows, cols, diagonals)
    constraints: List[BinaryConstraint] = []
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append(rows_cols_diags_constraint(variables[i], variables[j]))

    rng = random.Random(42)  # optional for reproducibility

    # Build and solve
    problem = DCSPProblem(variables, domains, constraints)
    sol = solve(
        problem,
        timeout_s=timeout_s,
        domain_reshuffling=True,
        rng=rng,
        reshuffle_iterations=150,   # single knob; <=0 means no per-run cap
        prefilter_domain=True,      # enable AC-3 pruning before each run
        verbosity=Verbosity.TQDM    # tqdm desc-only (if available), or quiet
    )

    if sol is None:
        print("No solution (or timeout).")
        return

    # Pretty-print a board
    grid = [["." for _ in range(N)] for _ in range(N)]
    for q, (r, c) in sol.items():
        grid[int(r)][int(c)] = "Q"
    print("\n".join(" ".join(row) for row in grid))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-Queens (2D-domain) with PyAsyncBTrack (ABT)")
    parser.add_argument("-n", "--size", type=int, default=8, help="Board size N")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout seconds (<=0 for unlimited)")
    args = parser.parse_args()
    main(N=args.size, timeout_s=args.timeout)
```

---

## Why “Asynchronous Backtracking”?

This package implements **ABT semantics** (nogoods, backjumping, asynchronous “agent” view) in a **single-process, centralized** solver that’s easy to embed. You get ABT’s powerful conflict learning without having to stand up a distributed system or message bus.

---

## Examples

This repo ships with two practical demos:

### 1) Latin Square (N × N)

```bash
python examples/latin_square_demo.py --n 4 --verbosity TQDM
python examples/latin_square_demo.py --n 5 --k 3 --solutions-timeout 5 --verbosity LOG
python examples/latin_square_demo.py --n 4 --givens "0,0=1; 1,1=2" --verbosity OFF
```

What it shows:
- Variables = grid cells, domains = symbols (e.g. `1..N` or `A..D`)
- Row/column **AllDifferent** via pairwise `!=`
- Optional **givens** as unary constraints
- Single solution or **multi-solution** enumeration

### 2) N-Queens (2D domain)

Values are `(row, col)` tuples; constraints enforce no shared rows/cols/diagonals.

```bash
python examples/example_NQueens.py -n 10 --timeout 120
python examples/example_NQueens_multiple_solutions.py -n 8 --timeout 120
```

What it shows:
- **2D domains** (any queen can occupy any cell)
- Pairwise constraints using a custom predicate
- Optional AC-3 pre-filtering and progress reporting
- Collect several **distinct** solutions

---

## Modeling DCSPs

### Concepts

- **Variables**: identifiers like `"X"`, `"Q0"`, `"X_0_1"`
- **Domains**: lists of values (ints, strings, tuples, frozensets)
- **Binary constraints**: relations over pairs `(u, v)` via fast, pure predicates

### Building a problem

```python
from pyasyncbtrack import DCSPProblem
from pyasyncbtrack.constraints import not_equal, alldifferent

variables = ["A", "B", "C"]
domains = {"A": [1,2], "B": [1,2], "C": [1,2]}

constraints = []
constraints += alldifferent(variables)  # expands to pairwise !=

problem = DCSPProblem(variables, domains, constraints)
```

### Common constraints

```python
from pyasyncbtrack.constraints import (
    eq, ne, lt, le, gt, ge,
    equals_offset, difference_ge,
    in_collection, not_in_collection, in_range,
    str_equals, str_not_equals, str_contains,
    alldifferent, allequal, monotone_increasing
)

# u != v
ne("X", "Y")

# |u - v| >= k
difference_ge("X", "Y", 2)

# X in {1,3,5} (paired against any neighbor)
in_collection("X", {1,3,5})("Y")
```

### Unary constraints (domain filters)

```python
from pyasyncbtrack.types import UnaryConstraint, apply_unary

domains = {"X": list(range(10))}
unaries = [UnaryConstraint("X", allowed=lambda v: v % 2 == 0)]
domains = apply_unary(domains, unaries)   # keeps only even values
```

---

## Solving

```python
from pyasyncbtrack import solve, Verbosity

result = solve(
    problem,
    timeout_s=20.0,             # None or <=0 means unlimited
    reshuffle_iterations=50_000,# per-run iteration cap (enables restarts)
    prefilter_domain=True,      # AC-3 before each run
    verbosity=Verbosity.TQDM,   # OFF | LOG | TQDM
    seed=7,                     # or pass rng=Random(...)
    # Enumeration (optional):
    nr_of_solutions=10,         # collect up to k distinct solutions
    solutions_timeout_s=60.0,   # enumeration time budget (seconds)
)
```

### Return shape

- **Single-solution mode**: returns `Assignment` (`dict[var] = value`) or `None`.
- **Enumeration mode** (`nr_of_solutions` set or `solutions_timeout_s` set): returns `List[Assignment]` (possibly empty).

---

## Configuration Reference

| Argument | Type | Default | Description |
|---|---|---:|---|
| `timeout_s` | `float \| None` | `10.0` | Global wall-clock budget for the whole call. |
| `use_mrv` | `bool` | `True` | **Minimum Remaining Values** variable selection. |
| `use_lcv` | `bool` | `True` | **Least Constraining Value** ordering. |
| `domain_reshuffling` | `bool` | `True` | Shuffle domains per run to diversify search. |
| `random_tiebreak` | `bool` | `True` | Jitter to break ties in selection/ordering. |
| `rng` | `random.Random \| None` | `None` | Provide your RNG (overrides `seed`). |
| `seed` | `int \| None` | `None` | Seed for deterministic runs (when `rng` not provided). |
| `reshuffle_iterations` | `int \| None` | `None` | Per-run iteration cap; triggers **restarts** when hit. |
| `prefilter_domain` | `bool` | `False` | Run **AC-3** before each run. |
| `verbosity` | `Verbosity` | `OFF` | `OFF`, `LOG`, or `TQDM` (desc-only). |
| `nr_of_solutions` | `int \| None` | `None` | Enumerate up to **k** unique solutions. |
| `solutions_timeout_s` | `float \| None` | `None` | Enumeration time budget (wall-clock). |
| `progress_log_every` | `int` | `5000` | LOG cadence (iterations). |
| `diversify_restarts` | `bool` | `True` | Per-run RNG diversification for broader exploration. |

---

## Tips & Best Practices

- **Domains matter**: narrow them early with unary constraints or AC-3 (`prefilter_domain=True`).
- **Heuristics**: keep MRV & LCV on for most problems.
- **Restarts**: for tough instances, set a per-run cap (`reshuffle_iterations`) and a sensible `timeout_s`.
- **Determinism**: pass a fixed `seed` (or an explicit `random.Random`) to reproduce results.
- **Enumeration**: use `nr_of_solutions` and/or `solutions_timeout_s`; solutions are **canonicalized** to avoid duplicates.

---

## API Surface (import paths)

```python
# Core
from pyasyncbtrack import DCSPProblem, solve, Verbosity

# Types & utilities
from pyasyncbtrack.types import (
    BinaryConstraint, UnaryConstraint, TableConstraint,
    apply_unary, Assignment, Variable, Value
)

# Reusable constraints
from pyasyncbtrack.constraints import (
    not_equal, equals, less_than, less_equal, greater_than, greater_equal,
    equals_offset, difference_ge, difference_gt, difference_le, difference_lt,
    in_collection, not_in_collection, in_range,
    str_equals, str_not_equals, str_has_prefix, str_has_suffix, str_contains,
    alldifferent, allequal, monotone_increasing, monotone_non_decreasing,
    equals_with_offset_chain, no_overlap, precedes, follows,
    pair,  # wrap custom (value,value) predicate quickly
)

# Consistency (optional)
from pyasyncbtrack.consistency import ac3
```

---

## CLI Demos

Run from the repository root:

```bash
# Latin squares
python examples/latin_square_demo.py --n 4 --verbosity TQDM

# N-Queens (2D domain)
python examples/example_NQueens.py -n 10 --timeout 120 TQDM

# N-Queens (2D domain) multiple solutions
python examples/example_NQueens_multiple_solutions.py -n 8 --timeout 120 TQDM
```

---

## Performance Notes

- Constraint predicates are in hot loops. Keep them **pure** and **fast**.
- If you write custom constraints, avoid expensive Python objects in inner calls.
- AC-3 can dramatically shrink domains for tight relations; for loose `!=` on large domains, its effect may be modest — test both ways.

---

## Python & Typing

- **Python**: 3.9+ recommended
- **Typing**: The public API is type-annotated and works well with Pyright/MyPy.

---

## License

This project is open source. See `LICENSE` in the repository for details.

---

## Acknowledgements

Inspired by the Asynchronous Backtracking literature and classic CSP propagation techniques (AC-3, MRV/LCV, nogoods, backjumping).

---

