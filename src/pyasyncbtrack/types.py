# src/pyasyncbtrack/types.py
from __future__ import annotations

"""
Types and helpers for modeling Distributed CSPs (DCSPs) in pyasyncbtrack.

This module offers:
- Fundamental type aliases: Variable, Value, Assignment
- BinaryConstraint: the native constraint unit for ABT
- Higher-level constraint types:
    * UnaryConstraint (domain filters)
    * TableConstraint (extensional binary constraints)
    * NaryConstraint (arbitrary predicate over many variables)
- Global-constraint helpers (e.g., AllDifferent) with safe decompositions
- Utilities to apply unary constraints and to compile/decompose constraints
  into ABT-ready binary constraints.

Notes
-----
ABT (Asynchronous Backtracking) operates over **binary constraints**. We still
support unary/n-ary/global constraints for modeling convenience and provide
helpers to:
  * Apply unary constraints by pruning domains BEFORE solving.
  * Decompose common global constraints (like AllDifferent) into a set of
    BinaryConstraint instances (e.g., pairwise !=).

If you need a custom n-ary predicate, you can either:
  - Decompose it yourself into binary predicates and pass those in, or
  - Wrap it as `NaryConstraint` and use `decompose_nary(...)` to produce
    an approximate or exact set of binary constraints (you control the logic).
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Set


# -----------------------------------------------------------------------------
# Core aliases
# -----------------------------------------------------------------------------

Variable = str
Value = int | str | float | Tuple | frozenset
Assignment = Dict[Variable, Value]

# Binary constraint predicate signature for ABT
BinaryPredicate = Callable[[Variable, Value, Variable, Value], bool]

# -----------------------------------------------------------------------------
# Binary constraint (ABT-native)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BinaryConstraint:
    """
    A binary constraint over two variables for ABT/DCSP.

    Parameters
    ----------
    u, v : Variable
        Variable identifiers. **Order matters** only if your predicate is not symmetric.
    predicate : BinaryPredicate
        Callable `(u_name, u_value, v_name, v_value) -> bool` that returns True
        iff the pair `(u = u_value, v = v_value)` is allowed.

    Contract
    --------
    - **Purity:** The predicate must be a pure function (no side-effects, no I/O).
    - **Determinism:** Given the same inputs, the predicate must return the same result.
    - **Symmetry (optional):** If the relation is logically symmetric, ensure
      `predicate(u, vu, v, vv) == predicate(v, vv, u, vu)`. ABT does not enforce this.

    Notes
    -----
    - Predicates receive *variable names* as well as *values*. This allows you to
      write name-aware constraints if needed, but most constraints should depend
      only on values for portability.
    - Keep predicates **fast**—they are called frequently during search.

    Examples
    --------
    >>> # Inequality over values
    >>> neq = BinaryConstraint("X", "Y", lambda ui, vi, vj, vjv: vi != vjv)
    >>> neq.predicate("X", 1, "Y", 2)
    True
    >>> neq.predicate("X", 3, "Y", 3)
    False

    >>> # Name-aware diagonal check (N-Queens-style, vars like 'Q0','Q1', values are columns)
    >>> diag = BinaryConstraint("Q0", "Q1",
    ...     lambda ui, vi, vj, vjv: abs(int(ui[1:]) - int(vj[1:])) != abs(vi - vjv))
    """
    u: Variable
    v: Variable
    predicate: BinaryPredicate


# -----------------------------------------------------------------------------
# Additional constraint types (modeling convenience)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class UnaryConstraint:
    """
    A unary constraint (domain filter) on a single variable.

    Parameters
    ----------
    var : Variable
        The variable to filter.
    allowed : Callable[[Value], bool]
        Predicate over values. Keep it pure and fast.

    Behavior
    --------
    Use `apply_unary(...)` to prune domains **before** solving. Unary constraints
    are not carried into ABT—they are applied once to reduce the search space.

    Examples
    --------
    >>> # Keep only even columns for Q0
    >>> UnaryConstraint("Q0", allowed=lambda x: x % 2 == 0)
    >>> # Restrict color choices
    >>> UnaryConstraint("color_A", allowed=lambda c: c in {"red", "blue"})

    Pitfalls
    --------
    - If your filter removes **all** values for `var`, the instance becomes UNSAT.
    - Avoid stateful checks; they break reproducibility and caching.
    """
    var: Variable
    allowed: Callable[[Value], bool]


@dataclass(frozen=True)
class TableConstraint:
    """
    Extensional (table) constraint for a binary relation.

    Parameters
    ----------
    u, v : Variable
        Variable identifiers.
    allowed_pairs : Set[Tuple[Value, Value]]
        Explicitly allowed (u_value, v_value) pairs.

    When to use
    -----------
    - The relation is small and easiest to specify by enumeration.
    - You want O(1) membership checks without writing a predicate.

    Examples
    --------
    >>> allowed = {(1, 2), (2, 3)}
    >>> t = TableConstraint("X", "Y", allowed_pairs=allowed)
    >>> (1, 2) in t.allowed_pairs
    True
    >>> (2, 2) in t.allowed_pairs
    False

    Notes
    -----
    - You can convert a `TableConstraint` into a `BinaryConstraint` with
      `compile_to_binary(...)`, which wraps the set membership as a predicate.
    - Keep the table modest in size; very large tables can hurt memory locality.
    """
    u: Variable
    v: Variable
    allowed_pairs: Set[Tuple[Value, Value]]


@dataclass(frozen=True)
class NaryConstraint:
    """
    N-ary constraint over an ordered list of variables.

    Parameters
    ----------
    vars : Sequence[Variable]
        Ordered variables this constraint relates.
    predicate : Callable[[Assignment], bool]
        Receives a *partial or full* assignment dict `{var: value}` and must return:
          - True  : consistent with current partial assignment
          - False : inconsistent (violated)

    ABT Compatibility
    -----------------
    ABT is **binary**. To use n-ary constraints with ABT you must:
    - Decompose them into a set of `BinaryConstraint`s (e.g., `alldifferent` → pairwise `!=`), or
    - Use a helper like `decompose_nary(...)` to generate a **sound** binary approximation.

    Examples
    --------
    >>> # Sum <= 10 over three variables (works with partial assignments)
    >>> def sum_leq_10(assn: Assignment) -> bool:
    ...     s = 0
    ...     for k in ("X", "Y", "Z"):
    ...         if k in assn:
    ...             s += assn[k]
    ...     return s <= 10
    >>> NaryConstraint(vars=("X","Y","Z"), predicate=sum_leq_10)

    Pitfalls
    --------
    - Decomposition can be expensive or weaker than the original global constraint.
    - Make your predicate **partial-friendly**: it will be called with incomplete
      assignments during decomposition and modeling utilities.
    """
    vars: Sequence[Variable]
    predicate: Callable[[Assignment], bool]



# -----------------------------------------------------------------------------
# Helper factories for common binary constraints
# -----------------------------------------------------------------------------

def not_equal(u: Variable, v: Variable) -> BinaryConstraint:
    """u != v (value inequality)."""
    return BinaryConstraint(u, v, lambda uu, vu, vv, vv_val: vu != vv_val)


def equals(u: Variable, v: Variable) -> BinaryConstraint:
    """u == v (value equality)."""
    return BinaryConstraint(u, v, lambda uu, vu, vv, vv_val: vu == vv_val)


def less_than(u: Variable, v: Variable) -> BinaryConstraint:
    """u < v (numeric)."""
    return BinaryConstraint(u, v, lambda uu, vu, vv, vv_val: (vu < vv_val))  # type: ignore[no-any-return]


def less_equal(u: Variable, v: Variable) -> BinaryConstraint:
    """u <= v (numeric)."""
    return BinaryConstraint(u, v, lambda uu, vu, vv, vv_val: (vu <= vv_val))  # type: ignore[no-any-return]


def greater_than(u: Variable, v: Variable) -> BinaryConstraint:
    """u > v (numeric)."""
    return BinaryConstraint(u, v, lambda uu, vu, vv, vv_val: (vu > vv_val))  # type: ignore[no-any-return]


def greater_equal(u: Variable, v: Variable) -> BinaryConstraint:
    """u >= v (numeric)."""
    return BinaryConstraint(u, v, lambda uu, vu, vv, vv_val: (vu >= vv_val))  # type: ignore[no-any-return]


def equals_offset(u: Variable, v: Variable, k: int | float) -> BinaryConstraint:
    """u == v + k (numeric offset)."""
    return BinaryConstraint(u, v, lambda uu, vu, vv, vv_val: vu == vv_val + k)


def difference_at_least(u: Variable, v: Variable, k: int | float) -> BinaryConstraint:
    """|u - v| >= k (numeric)."""
    return BinaryConstraint(u, v, lambda uu, vu, vv, vv_val: abs(vu - vv_val) >= k)  # type: ignore[arg-type]


def distance_not_equal(u: Variable, v: Variable, k: int | float) -> BinaryConstraint:
    """|u - v| != k (numeric)."""
    return BinaryConstraint(u, v, lambda uu, vu, vv, vv_val: abs(vu - vv_val) != k)  # type: ignore[arg-type]


def table(u: Variable, v: Variable, allowed_pairs: Iterable[Tuple[Value, Value]]) -> TableConstraint:
    """Construct a binary extensional (table) constraint."""
    return TableConstraint(u=u, v=v, allowed_pairs=set(allowed_pairs))


# -----------------------------------------------------------------------------
# Global constraint helpers (decompose to binary constraints)
# -----------------------------------------------------------------------------

def alldifferent(vars: Sequence[Variable]) -> List[BinaryConstraint]:
    """
    Decompose the AllDifferent(global) into pairwise not_equal constraints.

    This is the classic binary decomposition. For stronger propagation, one
    would use a dedicated global propagator (not in ABT scope). For ABT, pairwise
    != is the practical approach.
    """
    cs: List[BinaryConstraint] = []
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):
            cs.append(not_equal(vars[i], vars[j]))
    return cs


def monotone_increasing(vars: Sequence[Variable]) -> List[BinaryConstraint]:
    """
    Enforce vars[0] < vars[1] < ... < vars[n-1] using pairwise <.

    This is useful for ordering constraints or breaking symmetry.
    """
    cs: List[BinaryConstraint] = []
    for i in range(len(vars) - 1):
        cs.append(less_than(vars[i], vars[i + 1]))
    return cs


def equals_with_offset_chain(vars: Sequence[Variable], k: int | float) -> List[BinaryConstraint]:
    """
    Enforce a chain: v_i == v_{i+1} + k for all i.
    """
    cs: List[BinaryConstraint] = []
    for i in range(len(vars) - 1):
        cs.append(equals_offset(vars[i], vars[i + 1], k))
    return cs


# -----------------------------------------------------------------------------
# Utilities: applying unary constraints & compiling to ABT binaries
# -----------------------------------------------------------------------------

def apply_unary(domains: Dict[Variable, List[Value]], unary_constraints: Iterable[UnaryConstraint]) -> Dict[Variable, List[Value]]:
    """
    Apply unary constraints by filtering domains. Returns a NEW domains dict.

    IMPORTANT: Always call this BEFORE passing domains into the solver.
    """
    new_domains = {v: list(vals) for v, vals in domains.items()}
    by_var: Dict[Variable, List[UnaryConstraint]] = {}
    for uc in unary_constraints:
        by_var.setdefault(uc.var, []).append(uc)
    for var, filters in by_var.items():
        if var not in new_domains:
            raise KeyError(f"UnaryConstraint references unknown variable '{var}'")
        kept: List[Value] = []
        for val in new_domains[var]:
            if all(f.allowed(val) for f in filters):
                kept.append(val)
        new_domains[var] = kept
    return new_domains


def compile_to_binary(
    binaries: Iterable[BinaryConstraint] | None = None,
    tables: Iterable[TableConstraint] | None = None,
) -> List[BinaryConstraint]:
    """
    Merge native BinaryConstraints with TableConstraints converted to predicates.

    N-ary constraints are NOT handled here—decompose them separately.
    """
    out: List[BinaryConstraint] = []
    if binaries:
        out.extend(binaries)
    if tables:
        for t in tables:
            allowed = set(t.allowed_pairs)
            out.append(
                BinaryConstraint(
                    t.u,
                    t.v,
                    lambda uu, vu, vv, vv_val, _allowed=allowed: (vu, vv_val) in _allowed
                )
            )
    return out


def decompose_nary(
    nary: NaryConstraint,
    domains: Dict[Variable, List[Value]],
) -> List[BinaryConstraint]:
    """
    Example decomposition for an N-ary constraint into binary constraints.

    This function provides a **simple, conservative** scheme:
    - It will generate pairwise constraints that reject (val_i, val_j) pairs
      that **can never** appear in any global assignment satisfying `nary.predicate`.
    - It does so by testing the n-ary predicate on all **local** combinations
      of (val_i, val_j) for each pair (i, j) while other vars remain unassigned.
      If there exists *some* extension making the predicate True, the pair is kept.

    Warning
    -------
    - This can be expensive: O(sum over pairs of |D_i|*|D_j| * extension check)).
    - For large problems, implement a custom decomposition tailored to your constraint.

    Returns
    -------
    List[BinaryConstraint]
        A set of binary constraints that *soundly* approximates the n-ary constraint.
        (It will never forbid a pair that must be allowed in all solutions, but it can
         be weaker than the original.)
    """
    from itertools import product

    vars_seq = list(nary.vars)
    var_set = set(vars_seq)
    missing = var_set - set(domains.keys())
    if missing:
        raise KeyError(f"NaryConstraint references unknown variables: {sorted(missing)}")

    binaries: List[BinaryConstraint] = []

    for i in range(len(vars_seq)):
        for j in range(i + 1, len(vars_seq)):
            vi, vj = vars_seq[i], vars_seq[j]
            allowed_pairs: Set[Tuple[Value, Value]] = set()

            # Quick/weak check: Accept (ai, aj) if there exists an extension
            # of SOME subset of the remaining vars that satisfies the predicate.
            rest = [v for v in vars_seq if v not in (vi, vj)]
            # Heuristic: try an empty/partial assignment first (predicate should
            # handle partials). If not definitive, sample some combinations.
            for ai in domains[vi]:
                for aj in domains[vj]:
                    partial: Assignment = {vi: ai, vj: aj}
                    try:
                        ok_partial = nary.predicate(partial)
                    except Exception:
                        ok_partial = False
                    if ok_partial:
                        allowed_pairs.add((ai, aj))
                        continue
                    # Fallback: try small sampled extensions (first values)
                    # to avoid Cartesian explosion; feel free to specialize.
                    sample_ext = {r: (domains[r][0] if domains[r] else None) for r in rest}
                    if None in sample_ext.values():
                        # empty domain elsewhere; be conservative and skip
                        continue
                    trial = dict(partial, **sample_ext)  # shallow
                    try:
                        if nary.predicate(trial):
                            allowed_pairs.add((ai, aj))
                    except Exception:
                        # if predicate fails hard, treat as not allowed
                        pass

            binaries.append(
                BinaryConstraint(
                    vi,
                    vj,
                    lambda uu, vu, vv, vv_val, _allowed=allowed_pairs: (vu, vv_val) in _allowed
                )
            )

    return binaries
