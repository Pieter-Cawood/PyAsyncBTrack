# examples/latin_square_demo.py
from __future__ import annotations
"""
Latin Square as a generic DCSP using pyasyncbtrack (ABT).

Model
-----
- Variables: one variable per cell: X_r_c for r,c in {0..N-1}
- Domains: symbols (default 1..N)
- Constraints (binary only):
  * Row-wise: X_r_c1 != X_r_c2  for all c1 < c2
  * Col-wise: X_r1_c != X_r2_c  for all r1 < r2

Optionally, pass "givens" (prefilled cells) as unary constraints.

Run examples
------------
$ python examples/latin_square_demo.py --n 4 --verbosity TQDM
$ python examples/latin_square_demo.py --n 5 --k 3 --solutions-timeout 5 --verbosity LOG
$ python examples/latin_square_demo.py --n 4 --givens "0,0=1; 1,1=2" --verbosity OFF
"""

import argparse
from typing import Dict, List, Tuple, Optional

from pyasyncbtrack import Verbosity
from pyasyncbtrack.problem import DCSPProblem
from pyasyncbtrack.solver import solve
from pyasyncbtrack.constraints import not_equal
from pyasyncbtrack.types import UnaryConstraint, apply_unary, Assignment, Variable, Value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def var_name(r: int, c: int) -> str:
    return f"X_{r}_{c}"

def parse_givens(s: Optional[str]) -> Dict[Tuple[int, int], str]:
    """
    Parse e.g. '0,0=1; 1,2=3' into {(0,0): '1', (1,2): '3'}.
    Values are kept as strings so you can use arbitrary symbols.
    """
    if not s:
        return {}
    out: Dict[Tuple[int, int], str] = {}
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk or "," not in chunk:
            raise ValueError(f"Bad given: {chunk!r}. Use 'r,c=val'.")
        rc, val = chunk.split("=", 1)
        r_str, c_str = rc.split(",", 1)
        out[(int(r_str), int(c_str))] = val.strip()
    return out

def pretty_grid(sol: Assignment, n: int) -> str:
    # Detect width (supports multi-char symbols)
    width = 1
    for v in sol.values():
        width = max(width, len(str(v)))
    lines = []
    for r in range(n):
        row = [str(sol[var_name(r, c)]).rjust(width) for c in range(n)]
        lines.append(" ".join(row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DCSP builder for Latin Square
# ---------------------------------------------------------------------------

def build_latin_square_problem(
    n: int,
    symbols: Optional[List[str]] = None,
    givens: Optional[Dict[Tuple[int, int], str]] = None,
) -> Tuple[DCSPProblem, List[Variable], Dict[Variable, List[Value]]]:
    """
    Create a Latin-square DCSP.
    Returns (problem, variables, domains_used).
    """
    if symbols is None:
        # default symbols "1".."N" (strings so givens can be strings too)
        symbols = [str(i) for i in range(1, n + 1)]

    variables: List[str] = [var_name(r, c) for r in range(n) for c in range(n)]
    domains: Dict[str, List[Value]] = {v: list(symbols) for v in variables}

    # Constraints: row & column pairwise !=
    constraints = []
    # Rows
    for r in range(n):
        row_vars = [var_name(r, c) for c in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append(not_equal(row_vars[i], row_vars[j]))
    # Cols
    for c in range(n):
        col_vars = [var_name(r, c) for r in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append(not_equal(col_vars[i], col_vars[j]))

    # Apply unary givens (if any)
    if givens:
        unaries: List[UnaryConstraint] = []
        for (r, c), val in givens.items():
            v = var_name(r, c)
            if v not in domains:
                raise KeyError(f"Given references out-of-range cell {(r, c)}")
            unaries.append(UnaryConstraint(var=v, allowed=lambda x, val=val: str(x) == str(val)))
        domains = apply_unary(domains, unaries)

    problem = DCSPProblem(variables, domains, constraints)
    return problem, variables, domains


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Latin Square with pyasyncbtrack (ABT)")
    ap.add_argument("--n", type=int, default=4, help="Order of the Latin square.")
    ap.add_argument("--symbols", type=str, default=None,
                   help='Comma list of symbols, e.g. "A,B,C,D". Default: "1..N".')
    ap.add_argument("--givens", type=str, default=None,
                   help='Semicolon list of r,c=val (e.g. "0,0=1; 1,2=3").')

    ap.add_argument("--timeout", type=float, default=20.0, help="Global timeout (s).")
    ap.add_argument("--reshuffle-iterations", type=int, default=50_000,
                   help="Per-run iteration cap before restart; <=0 = no restarts.")
    ap.add_argument("--prefilter-domain", action="store_true",
                   help="Apply AC-3 arc-consistency before each run.")
    ap.add_argument("--verbosity", choices=["OFF", "LOG", "TQDM"], default="TQDM",
                   help="Iteration progress display.")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed.")

    # Multi-solution enumeration (optional)
    ap.add_argument("--k", type=int, default=None, help="Collect up to k distinct solutions.")
    ap.add_argument("--solutions-timeout", type=float, default=None,
                   help="Time budget for enumeration (seconds).")

    args = ap.parse_args()

    # Symbols
    sym_list = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    givens = parse_givens(args.givens)

    # Build problem
    problem, variables, _ = build_latin_square_problem(args.n, sym_list, givens)

    # Solve (kwargs interface)
    result = solve(
        problem,
        timeout_s=args.timeout,
        reshuffle_iterations=args.reshuffle_iterations,
        prefilter_domain=args.prefilter_domain,
        verbosity=Verbosity[args.verbosity],
        seed=args.seed,
        # enumeration (optional)
        nr_of_solutions=args.k,
        solutions_timeout_s=args.solutions_timeout,
    )

    # Normalize solutions
    sols: List[Assignment]
    if result is None:
        print("No solution (or timeout).")
        return
    elif isinstance(result, dict):
        sols = [result]
    else:
        sols = result

    # Print them
    print(f"Found {len(sols)} solution(s).")
    for i, sol in enumerate(sols, 1):
        print(f"\nSolution {i}:")
        print(pretty_grid(sol, args.n))


if __name__ == "__main__":
    main()
