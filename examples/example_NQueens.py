#examples/example_NQueens.py
from __future__ import annotations
"""
N-Queens as a generic DCSP using pyasyncbtrack, with a 2D domain.

Model
-----
- Variables: queens Q0..Q{N-1}
- Domain of each variable: all board cells (row, col) with row,col ∈ {0..N-1}
- Constraints (for every pair of queens):
  * Different rows          : r_i != r_j
  * Different columns       : c_i != c_j
  * Different diagonals     : |r_i - r_j| != |c_i - c_j|

Notes
-----
- This 2D domain allows *any* queen to occupy *any* row/column.
- Classic N-Queens solutions still require exactly one queen per row & column,
  which emerge from the pairwise constraints above.
"""

import argparse
import random

from typing import List, Dict, Optional, Tuple

from pyasyncbtrack import Verbosity
from pyasyncbtrack.problem import DCSPProblem
from pyasyncbtrack.solver import solve
from pyasyncbtrack.types import BinaryConstraint, UnaryConstraint, apply_unary, alldifferent


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
# Rendering
# ---------------------------------------------------------------------------

def pretty_board(cells: Dict[str, Tuple[int, int]], N: int) -> str:
    """
    Render a board from an assignment mapping queen -> (row, col).
    """
    grid = [["." for _ in range(N)] for _ in range(N)]
    for q, (r, c) in cells.items():
        if 0 <= r < N and 0 <= c < N:
            grid[r][c] = "Q"
    return "\n".join(" ".join(row) for row in grid)

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main(N: int = 8, even_first: bool = False, timeout_s: Optional[float] = 10.0) -> None:
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
        reshuffle_iterations=150,   # <- single knob; None/<=0 means no reshuffle cap
        prefilter_domain=True,                # <- enable AC-3 pruning before each run
        verbosity=Verbosity.TQDM         # tqdm desc-only; or single stderr line if tqdm not present
    )

    # Report
    if sol is None:
        print("No solution (or timeout).")
        return

    # Convert to queen -> (row, col)
    cells: Dict[str, Tuple[int, int]] = {q: tuple(sol[q]) for q in variables}  # type: ignore[assignment]
    print("Solution (queen -> (row, col)):", cells)
    print("Board:\n" + pretty_board(cells, N))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-Queens (2D-domain) with pyasyncbtrack (ABT)")
    parser.add_argument("-n", "--size", type=int, default=10, help="Board size N")
    parser.add_argument(
        "--even-first",
        default=False,  # now true means "apply the demo unary"
        type=bool,
        help="Restrict Q0 to even columns (demo of unary constraints)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Wall-clock timeout in seconds (default: 10.0; use 0/negative to wait forever)",
    )
    args = parser.parse_args()
    main(N=args.size, even_first=args.even_first, timeout_s=args.timeout)
