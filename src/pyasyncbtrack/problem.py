# src/pyasyncbtrack/problem.py
from __future__ import annotations

"""
DCSPProblem: Container for Distributed Constraint Satisfaction Problem instances.

What this module gives you
--------------------------
- A problem container that ABT agents use at runtime (variables, domains, constraints).
- **Generic domain support**: callers can pass any iterable (list/tuple/set/range/…);
  we normalize to lists internally for fast iteration.
- **Configurable solution shape** via a `solution_formatter` (dict by default, or
  list/tuple/custom objects).
- **Reset utilities** so you can reuse the same `DCSPProblem` across experiments
  (clear only outcome or also restore original domains).

Quick start
-----------
>>> from pyasyncbtrack.problem import (
...     DCSPProblem, solution_as_dict, solution_as_ordered_list, solution_as_tuple
... )
>>> variables = ["X", "Y", "Z"]
>>> domains = {"X": {1, 2}, "Y": range(1, 3), "Z": (1, 2)}  # any iterable works
>>> constraints = []  # fill with BinaryConstraint instances
>>> problem = DCSPProblem.from_domains(
...     variables, domains, constraints, solution_formatter=solution_as_tuple
... )
>>> # ... run the scheduler ...
>>> # After solving, problem.solution is a tuple (val(X), val(Y), val(Z)).

Notes
-----
- **Priority order** is defined by the order of `variables` (index 0 is highest).
- **Neighbors** are computed from the constraint endpoints and returned sorted by priority.
- This class is **mutable by design** for runtime outcome fields (`solved`, `solution`)
  and for resets; treat modeling inputs (`variables`, `domains`, `constraints`) as
  immutable by convention after construction.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
)

from .types import Variable, Value, BinaryConstraint


# -----------------------------------------------------------------------------
# Solution formatting API
# -----------------------------------------------------------------------------

SolutionFormatter = Callable[[Mapping[Variable, Value], Sequence[Variable]], Any]
"""
A function that converts a full assignment (var -> value) into whatever shape
the caller prefers. It receives:

- assignment: Mapping[Variable, Value]   # complete assignment
- variables:  Sequence[Variable]         # variable order (for ordered outputs)

and returns Any (e.g., dict, list, tuple, dataclass, custom record, ...).
"""


def solution_as_dict(
    assignment: Mapping[Variable, Value],
    variables: Sequence[Variable],
) -> Dict[Variable, Value]:
    """
    Default solution formatter: keep the assignment as a dict.

    Suitable when you want `{"X": 1, "Y": 2, ...}`.
    """
    return dict(assignment)


def solution_as_ordered_list(
    assignment: Mapping[Variable, Value],
    variables: Sequence[Variable],
) -> List[Value]:
    """
    Solution as a list of values in `variables` order.

    Example: variables=["X","Y"] -> [assignment["X"], assignment["Y"]]
    """
    return [assignment[v] for v in variables]


def solution_as_tuple(
    assignment: Mapping[Variable, Value],
    variables: Sequence[Variable],
) -> tuple[Value, ...]:
    """
    Solution as a tuple of values in `variables` order.

    Example: variables=["X","Y"] -> (assignment["X"], assignment["Y"])
    """
    return tuple(assignment[v] for v in variables)


# -----------------------------------------------------------------------------
# DCSPProblem
# -----------------------------------------------------------------------------

@dataclass  # not frozen: runtime outcome fields mutate; inputs treated as immutable by convention
class DCSPProblem:
    """
    DCSPProblem: Container for Distributed Constraint Satisfaction Problem instances.

    Overview
    --------
    A `DCSPProblem` defines the data structures that ABT agents operate on:
      - `variables`   : ordered sequence (defines global priority).
      - `domains`     : mapping var -> iterable of possible values (any collection
                        accepted via `from_domains`; normalized to lists internally).
      - `constraints` : sequence of `BinaryConstraint` relations.

    Runtime outcome
    ---------------
    - `solved`   : True iff a complete assignment was recorded.
    - `solution` : The *formatted* solution (dict/list/tuple/custom) or None.
    - `solution_formatter` : Controls how solutions are exposed.

    Construction
    ------------
    Prefer `DCSPProblem.from_domains(...)` if your domains are not already lists.
    If you call the constructor directly, `__post_init__` will still normalize
    domain collections to lists.

    Reset
    -----
    - `reset_outcome()` clears `solved/solution` only.
    - `reset(restore_domains=True)` also restores domains to their original snapshot.

    Tips
    ----
    - Keep predicates in constraints *pure* and *fast*.
    - Be mindful that reordering `variables` changes the priority scheme and may
      change search behavior.
    """

    # --- modeling inputs (canonical/internal representation) ---
    variables: Sequence[Variable]
    domains: Dict[Variable, List[Value]]  # normalized to list for fast iteration
    constraints: Sequence[BinaryConstraint]

    # --- runtime outcome (set by the scheduler or caller) ---
    solved: bool = field(default=False, init=False)
    solution: Optional[Any] = field(default=None, init=False)

    # --- output formatting & original domain snapshot (for reset) ---
    solution_formatter: SolutionFormatter = field(default=solution_as_dict, repr=False)
    _initial_domains: Dict[Variable, List[Value]] = field(default_factory=dict, init=False, repr=False)

    # -------------------------------------------------------------------------
    # Preferred constructor: accept generic domain collections
    # -------------------------------------------------------------------------
    @classmethod
    def from_domains(
        cls,
        variables: Sequence[Variable],
        domains: Mapping[Variable, Iterable[Value]],
        constraints: Sequence[BinaryConstraint],
        *,
        solution_formatter: SolutionFormatter = solution_as_dict,
    ) -> "DCSPProblem":
        """
        Build a DCSPProblem from domains provided as ANY iterable collections.

        All domain collections (list/tuple/set/range/…) are coerced to lists
        internally to ensure efficient, deterministic iteration.

        Parameters
        ----------
        variables : Sequence[Variable]
            Ordered variables (index = priority).
        domains : Mapping[Variable, Iterable[Value]]
            Any iterable per variable (e.g., set, range, tuple).
        constraints : Sequence[BinaryConstraint]
            Binary relations over pairs of variables.
        solution_formatter : SolutionFormatter, optional
            Controls the shape of `problem.solution` and of the scheduler's return.

        Returns
        -------
        DCSPProblem
            A problem instance with canonical list-based domains.
        """
        canonical = {v: list(vals) for v, vals in domains.items()}
        return cls(
            variables=variables,
            domains=canonical,
            constraints=constraints,
            solution_formatter=solution_formatter,
        )

    # -------------------------------------------------------------------------
    # Lifecycle hooks
    # -------------------------------------------------------------------------
    def __post_init__(self) -> None:
        """
        Normalize incoming domains and snapshot an original copy.

        - Ensures every domain value collection is a `list`.
        - Stores `_initial_domains` for later restoration via `reset(restore_domains=True)`.
        """
        # Normalize any non-list domains to lists (defensive if constructor used directly)
        self.domains = {v: (vals if isinstance(vals, list) else list(vals)) for v, vals in self.domains.items()}
        # Snapshot original domains for optional restoration
        self._initial_domains = {v: list(vals) for v, vals in self.domains.items()}

    # -------------------------------------------------------------------------
    # Priority
    # -------------------------------------------------------------------------
    def priority(self, var: Variable) -> int:
        """
        Return the global priority rank of a variable.

        Lower index in `variables` ⇒ higher priority.
        """
        return list(self.variables).index(var)

    # -------------------------------------------------------------------------
    # Neighborhood
    # -------------------------------------------------------------------------
    def neighbors(self, var: Variable) -> List[Variable]:
        """
        Return neighbor variables of `var` (those that share at least one constraint),
        sorted by global priority.
        """
        nbs: set[Variable] = set()
        for c in self.constraints:
            if c.u == var:
                nbs.add(c.v)
            elif c.v == var:
                nbs.add(c.u)
        return sorted(nbs, key=self.priority)

    # -------------------------------------------------------------------------
    # Runtime helpers (used by scheduler / caller)
    # -------------------------------------------------------------------------
    def set_solution_formatter(self, formatter: SolutionFormatter) -> None:
        """
        Change how solutions are exposed via `self.solution` (and returned by the scheduler).

        Examples
        --------
        >>> problem.set_solution_formatter(solution_as_ordered_list)
        >>> problem.set_solution_formatter(solution_as_tuple)
        >>> # Custom record:
        >>> problem.set_solution_formatter(lambda a, vs: {"vars": list(vs), "vals": [a[v] for v in vs]})
        """
        self.solution_formatter = formatter

    def mark_solved(self, assignment: Mapping[Variable, Value]) -> None:
        """
        Record a complete, consistent assignment as the solution (formatted).
        Validates:
          - Coverage (no unknown variables, none missing)
          - Domain membership (value ∈ domain[var])
          - **All binary constraints hold**
        """
        # Coverage checks
        unknown = set(assignment.keys()) - set(self.variables)
        if unknown:
            raise ValueError(f"Assignment contains unknown variables: {sorted(unknown)}")
        missing = [v for v in self.variables if v not in assignment]
        if missing:
            raise ValueError(f"Assignment missing variables: {missing}")

        # Simple domain checks (lightweight guard)
        for v, val in assignment.items():
            if v not in self.domains:
                raise ValueError(f"Variable {v!r} missing from domains.")
            if val not in self.domains[v]:
                raise ValueError(f"Value {val!r} not in domain of {v!r}: {self.domains[v]!r}")

        # Constraint checks
        for c in self.constraints:
            u, v = c.u, c.v
            try:
                ok = c.predicate(u, assignment[u], v, assignment[v])
            except Exception as e:
                raise ValueError(f"Constraint predicate raised on {u},{v}: {e}") from e
            if not ok:
                raise ValueError(
                    f"Assignment violates constraint {u}-{v}: {u}={assignment[u]!r}, {v}={assignment[v]!r}"
                )

        # Format and commit
        formatted = self.solution_formatter(assignment, self.variables)
        self.solved = True
        self.solution = formatted

    def reset_outcome(self) -> None:
        """
        Clear runtime outcome fields only.

        - Leaves `domains` untouched (useful if you have already applied
          unary pruning and want to keep it).
        """
        self.solved = False
        self.solution = None

    def reset(self, *, restore_domains: bool = False) -> None:
        """
        Reset the problem for a fresh solve.

        Parameters
        ----------
        restore_domains : bool, default False
            If True, restore `domains` to the snapshot taken at construction.
            If False, keep current domains (e.g., after `apply_unary` pruning).

        Notes
        -----
        - Variables and constraints are not mutated.
        - This also calls `reset_outcome()` to clear any prior solution.
        """
        self.reset_outcome()
        if restore_domains:
            self.domains = {v: list(vals) for v, vals in self._initial_domains.items()}
