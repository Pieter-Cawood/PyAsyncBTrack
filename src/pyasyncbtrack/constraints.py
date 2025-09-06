# src/pyasyncbtrack/constraints.py
"""
Common reusable binary constraints and helper factories.

This module provides convenience constructors for frequently used
binary constraints in DCSP/ABT problems, plus small utilities to
generate families of constraints (e.g., AllDifferent / AllEqual).

All helpers return :class:`BinaryConstraint` instances bound to specific
variable names unless otherwise noted.

Coverage
--------
- Generic comparisons: ==, !=, <, <=, >, >= (with short aliases)
- Offsets & distances: u == v + k, |u - v| {<,<=,>,>=,!=,==} k
- Membership: in_collection / not_in_collection (per-variable)
- Group helpers: alldifferent, allequal, monotone chains, equals-with-offset chains
- Scheduling / intervals: non-overlap, precedes/follows (optional; can remove if not needed)
- Strings: equality/inequality, prefix/suffix/contains
- Custom: wrap any (value, value) predicate via `pair(...)`

Notes
-----
- Keep predicates **pure** and **fast**—they are called frequently during search.
- If your problem requires custom predicates, wrap them using `pair(...)`.
"""

from __future__ import annotations
from typing import Callable, Iterable, List, Sequence, Set, Tuple, Any

from .types import BinaryConstraint, Variable, Value


# -----------------------------------------------------------------------------
# Core helper
# -----------------------------------------------------------------------------

def pair(u: Variable, v: Variable, pred: Callable[[Value, Value], bool]) -> BinaryConstraint:
    """
    Wrap a simple value-level predicate into a `BinaryConstraint`.

    Parameters
    ----------
    u, v : Variable
        Variable identifiers.
    pred : Callable[[Value, Value], bool]
        A binary predicate over values. Must return True iff (u=val_u, v=val_v) is allowed.

    Returns
    -------
    BinaryConstraint
        A constraint object suitable for ABT.

    Example
    -------
    >>> c = pair("X", "Y", lambda a, b: a < b)
    >>> c.predicate("X", 1, "Y", 2)
    True
    >>> c.predicate("X", 2, "Y", 1)
    False
    """
    return BinaryConstraint(u=u, v=v, predicate=lambda ui, vi, vj, vjv: pred(vi, vjv))


# -----------------------------------------------------------------------------
# Generic numeric/value comparisons (+ aliases)
# -----------------------------------------------------------------------------

def equals(u: Variable, v: Variable) -> BinaryConstraint:
    """Enforce equality: u == v."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: vi == vjv)

def not_equal(u: Variable, v: Variable) -> BinaryConstraint:
    """Enforce inequality: u != v."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: vi != vjv)

def less_than(u: Variable, v: Variable) -> BinaryConstraint:
    """Enforce numeric comparison: u < v."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: vi < vjv)  # type: ignore[no-any-return]

def less_equal(u: Variable, v: Variable) -> BinaryConstraint:
    """Enforce numeric comparison: u <= v."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: vi <= vjv)  # type: ignore[no-any-return]

def greater_than(u: Variable, v: Variable) -> BinaryConstraint:
    """Enforce numeric comparison: u > v."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: vi > vjv)  # type: ignore[no-any-return]

def greater_equal(u: Variable, v: Variable) -> BinaryConstraint:
    """Enforce numeric comparison: u >= v."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: vi >= vjv)  # type: ignore[no-any-return]

# Short aliases (ergonomic)
eq = equals
ne = not_equal
lt = less_than
le = less_equal
gt = greater_than
ge = greater_equal


# -----------------------------------------------------------------------------
# Offsets & distance-based constraints (numeric)
# -----------------------------------------------------------------------------

def equals_offset(u: Variable, v: Variable, k: int | float) -> BinaryConstraint:
    """Enforce u == v + k."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: vi == vjv + k)

def difference_eq(u: Variable, v: Variable, k: int | float) -> BinaryConstraint:
    """Enforce |u - v| == k."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: abs(vi - vjv) == k)  # type: ignore[arg-type]

def difference_ne(u: Variable, v: Variable, k: int | float) -> BinaryConstraint:
    """Enforce |u - v| != k."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: abs(vi - vjv) != k)  # type: ignore[arg-type]

def difference_ge(u: Variable, v: Variable, k: int | float) -> BinaryConstraint:
    """Enforce |u - v| >= k."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: abs(vi - vjv) >= k)  # type: ignore[arg-type]

def difference_gt(u: Variable, v: Variable, k: int | float) -> BinaryConstraint:
    """Enforce |u - v| > k."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: abs(vi - vjv) > k)  # type: ignore[arg-type]

def difference_le(u: Variable, v: Variable, k: int | float) -> BinaryConstraint:
    """Enforce |u - v| <= k."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: abs(vi - vjv) <= k)  # type: ignore[arg-type]

def difference_lt(u: Variable, v: Variable, k: int | float) -> BinaryConstraint:
    """Enforce |u - v| < k."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: abs(vi - vjv) < k)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Membership / collection helpers (per-variable)
# -----------------------------------------------------------------------------
# These return factories because ABT needs BinaryConstraint (u,v, predicate).
# The predicate ignores v's value; it only constrains u's value.

def in_collection(u: Variable, allowed: Iterable[Value]) -> Callable[[Variable], BinaryConstraint]:
    """
    Enforce value(u) ∈ allowed (ignoring v).

    Usage
    -----
    >>> # X ∈ {1,2,3}, pair against any neighbor Y
    >>> cX = in_collection("X", {1,2,3})
    >>> constraint = cX("Y")
    """
    allowed_set = set(allowed)
    def _factory(v: Variable) -> BinaryConstraint:
        return BinaryConstraint(u, v, lambda ui, vi, vj, vjv, _A=allowed_set: vi in _A)
    return _factory

def not_in_collection(u: Variable, forbidden: Iterable[Value]) -> Callable[[Variable], BinaryConstraint]:
    """Enforce value(u) ∉ forbidden (ignoring v)."""
    forbidden_set = set(forbidden)
    def _factory(v: Variable) -> BinaryConstraint:
        return BinaryConstraint(u, v, lambda ui, vi, vj, vjv, _F=forbidden_set: vi not in _F)
    return _factory

def in_range(u: Variable, lo: Value, hi: Value, inclusive: bool = True) -> Callable[[Variable], BinaryConstraint]:
    """
    Enforce value(u) within [lo, hi] if inclusive else (lo, hi).

    Works for numeric or orderable values.
    """
    if inclusive:
        def _pred(val: Value) -> bool:  # type: ignore[no-redef]
            return lo <= val <= hi
    else:
        def _pred(val: Value) -> bool:  # type: ignore[no-redef]
            return lo < val < hi

    def _factory(v: Variable) -> BinaryConstraint:
        return BinaryConstraint(u, v, lambda ui, vi, vj, vjv, _p=_pred: _p(vi))
    return _factory


# -----------------------------------------------------------------------------
# String-oriented helpers (when values are strings)
# -----------------------------------------------------------------------------

def str_equals(u: Variable, v: Variable) -> BinaryConstraint:
    """Enforce string equality of values: u == v."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: str(vi) == str(vjv))

def str_not_equals(u: Variable, v: Variable) -> BinaryConstraint:
    """Enforce string inequality of values: u != v."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: str(vi) != str(vjv))

def str_has_prefix(u: Variable, v: Variable) -> BinaryConstraint:
    """Enforce value(u) is a prefix of value(v)."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: str(vjv).startswith(str(vi)))

def str_has_suffix(u: Variable, v: Variable) -> BinaryConstraint:
    """Enforce value(u) is a suffix of value(v)."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: str(vjv).endswith(str(vi)))

def str_contains(u: Variable, v: Variable) -> BinaryConstraint:
    """Enforce value(u) is a substring of value(v)."""
    return BinaryConstraint(u, v, lambda ui, vi, vj, vjv: str(vi) in str(vjv))


# -----------------------------------------------------------------------------
# Group helpers (return lists of BinaryConstraint)
# -----------------------------------------------------------------------------

def alldifferent(vars: Sequence[Variable]) -> List[BinaryConstraint]:
    """
    Decompose AllDifferent(vars) into pairwise u != v constraints.

    Returns a list of `BinaryConstraint` instances covering every pair.
    """
    cs: List[BinaryConstraint] = []
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):
            cs.append(not_equal(vars[i], vars[j]))
    return cs

def allequal(vars: Sequence[Variable]) -> List[BinaryConstraint]:
    """
    Enforce all variables have the same value: v0 == v1 == ... == v_{n-1}.

    Implemented as pairwise equalities between consecutive variables
    to keep the number of constraints linear.
    """
    cs: List[BinaryConstraint] = []
    for i in range(len(vars) - 1):
        cs.append(equals(vars[i], vars[i + 1]))
    return cs

def monotone_increasing(vars: Sequence[Variable]) -> List[BinaryConstraint]:
    """Enforce vars[0] < vars[1] < ... < vars[n-1]."""
    cs: List[BinaryConstraint] = []
    for i in range(len(vars) - 1):
        cs.append(less_than(vars[i], vars[i + 1]))
    return cs

def monotone_non_decreasing(vars: Sequence[Variable]) -> List[BinaryConstraint]:
    """Enforce vars[0] <= vars[1] <= ... <= vars[n-1]."""
    cs: List[BinaryConstraint] = []
    for i in range(len(vars) - 1):
        cs.append(less_equal(vars[i], vars[i + 1]))
    return cs

def equals_with_offset_chain(vars: Sequence[Variable], k: int | float) -> List[BinaryConstraint]:
    """Enforce a chain: v_i == v_{i+1} + k for all i."""
    cs: List[BinaryConstraint] = []
    for i in range(len(vars) - 1):
        cs.append(equals_offset(vars[i], vars[i + 1], k))
    return cs


# -----------------------------------------------------------------------------
# Optional: Scheduling / intervals (remove if out of scope)
# -----------------------------------------------------------------------------

def no_overlap(u: Variable, v: Variable) -> BinaryConstraint:
    """
    Enforce that two intervals do not overlap.

    Assumes values are `(start, end)` tuples with start < end.
    """
    def _pred(ui: str, vi: Tuple[Any, Any], vj: str, vjv: Tuple[Any, Any]) -> bool:
        s1, e1 = vi
        s2, e2 = vjv
        return e1 <= s2 or e2 <= s1
    return BinaryConstraint(u, v, _pred)

def precedes(u: Variable, v: Variable, gap: int | float = 0) -> BinaryConstraint:
    """Enforce end(u) + gap <= start(v) for interval values (start, end)."""
    def _pred(ui: str, vi: Tuple[Any, Any], vj: str, vjv: Tuple[Any, Any]) -> bool:
        _, e1 = vi
        s2, _ = vjv
        return e1 + gap <= s2
    return BinaryConstraint(u, v, _pred)

def follows(u: Variable, v: Variable, gap: int | float = 0) -> BinaryConstraint:
    """Enforce end(v) + gap <= start(u) for interval values (start, end)."""
    def _pred(ui: str, vi: Tuple[Any, Any], vj: str, vjv: Tuple[Any, Any]) -> bool:
        _, e2 = vjv
        s1, _ = vi
        return e2 + gap <= s1
    return BinaryConstraint(u, v, _pred)
