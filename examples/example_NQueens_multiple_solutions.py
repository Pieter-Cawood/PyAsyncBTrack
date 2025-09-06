#examples/example_NQueens_multiple_solutions.py
from __future__ import annotations
import argparse
import random
import sys
from typing import List, Dict, Optional, Tuple

from pyasyncbtrack import Verbosity
from pyasyncbtrack.problem import DCSPProblem
from pyasyncbtrack.solver import solve
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
    return BinaryConstraint(u, v, pred)

def validate_queens(cells: Dict[str, Tuple[int, int]], N: int) -> None:
    """Raise AssertionError if not exactly one per row/column."""
    rows = [0]*N
    cols = [0]*N
    for _, (r, c) in cells.items():
        rows[r] += 1
        cols[c] += 1
    assert all(x == 1 for x in rows), f"Row counts invalid: {rows}"
    assert all(x == 1 for x in cols), f"Col counts invalid: {cols}"

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def main(N: int = 8, timeout_s: Optional[float] = 10.0) -> None:
    variables = [f"Q{i}" for i in range(N)]
    all_cells: List[Tuple[int, int]] = [(r, c) for r in range(N) for c in range(N)]
    domains: Dict[str, List[Tuple[int, int]]] = {q: list(all_cells) for q in variables}

    constraints: List[BinaryConstraint] = []
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append(rows_cols_diags_constraint(variables[i], variables[j]))

    problem = DCSPProblem(variables, domains, constraints)

    sols = solve(
        problem,
        nr_of_solutions=10,
        solutions_timeout_s=130.0,
        seed=7,
        prefilter_domain=True,
        reshuffle_iterations=200,
        verbosity=Verbosity.TQDM,
    )

    # If tqdm left a live line around, finish it cleanly before printing boards.
    sys.stdout.flush()
    sys.stderr.flush()
    print()  # ensure a clean line break after tqdm

    if not sols:
        print("No solution (or timeout).")
        return

    # Use a canonical variable order so coordinate listing + printing are stable.
    ordered_vars = sorted(variables, key=lambda s: (len(s), s))

    for k, sol in enumerate(sols, 1):
        # Make a safe, typed mapping queen -> (row, col)
        cells: Dict[str, Tuple[int, int]] = {
            q: (int(sol[q][0]), int(sol[q][1])) for q in ordered_vars
        }

        # Sanity check: exactly one per row & column
        try:
            validate_queens(cells, N)
        except AssertionError as e:
            print(f"[warn] solution {k} failed validation: {e}")

        print(f"\nSolution {k} (queen -> (row, col)):")
        print(cells)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-Queens (2D-domain) with pyasyncbtrack (ABT)")
    parser.add_argument("-n", "--size", type=int, default=8, help="Board size N")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Wall-clock timeout in seconds (default: 10.0; use 0/negative to wait forever)",
    )
    args = parser.parse_args()
    main(N=args.size, timeout_s=args.timeout)
