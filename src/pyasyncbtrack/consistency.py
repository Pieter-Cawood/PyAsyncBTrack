# src/pyasyncbtrack/consistency.py
from __future__ import annotations
"""
Arc Consistency (AC-3) for binary DCSPs.

This module provides a generic AC-3 implementation operating over:
- problem.variables
- problem.domains: Dict[Variable, List[Value]]
- problem.constraints: Iterable[BinaryConstraint] (predicate(u, vu, v, vv) -> bool)
- optional problem.neighbors(var)

Use `ac3(problem, domains=None)` to prune domains *before* solving.
It returns a NEW domains dict (unless in_place=True). Returns None if a wipeout occurs.

Notes
-----
- AC-3 enforces: for every arc (Xi, Xj), for each a ∈ Di there exists b ∈ Dj
  with which all constraints between Xi and Xj are satisfied.
- For loose constraints like simple inequality on wide domains, AC-3 may prune little.
  For tight/extensional constraints it can substantially shrink the search space.
"""

from typing import Dict, List, Tuple, Iterable, Optional, Callable
from collections import deque
import logging

from .types import Variable, Value, BinaryConstraint

log = logging.getLogger("pyasyncbtrack.consistency")


def _neighbors_of(problem, var: Variable) -> List[Variable]:
    """Get neighbors(var) if available; otherwise derive from constraints."""
    try:
        return list(problem.neighbors(var))
    except Exception:
        ns: set[Variable] = set()
        for c in problem.constraints:
            if c.u == var:
                ns.add(c.v)
            elif c.v == var:
                ns.add(c.u)
        return list(ns)


def _constraints_between(problem, u: Variable, v: Variable) -> List[BinaryConstraint]:
    """Filter the global list to only those constraints that touch (u, v)."""
    out: List[BinaryConstraint] = []
    for c in problem.constraints:
        if (c.u == u and c.v == v) or (c.u == v and c.v == u):
            out.append(c)
    return out


def _revise(
    problem,
    u: Variable,
    v: Variable,
    domains: Dict[Variable, List[Value]],
    cons_uv: List[BinaryConstraint],
) -> Tuple[bool, int]:
    """
    REVISE step: remove from D(u) any a for which there is no support b ∈ D(v).

    Returns
    -------
    changed : bool
        True iff D(u) was reduced.
    removed_count : int
        Number of values removed from D(u).
    """
    Du = domains[u]
    Dv = domains[v]
    if not Du or not Dv:
        return False, 0

    keep: List[Value] = []
    removed = 0

    # For each a in Du, check if ∃ b in Dv s.t. all constraints allow (u=a, v=b).
    for a in Du:
        supported = False
        for b in Dv:
            ok = True
            for c in cons_uv:
                # Feed values in (c.u, c.v) order
                u_val = a if c.u == u else b
                v_val = b if c.v == v else a
                if not c.predicate(c.u, u_val, c.v, v_val):
                    ok = False
                    break
            if ok:
                supported = True
                break
        if supported:
            keep.append(a)
        else:
            removed += 1

    if removed:
        domains[u] = keep
        log.debug("AC3: revised %s wrt %s → removed %d (|Du|=%d→%d)", u, v, removed, len(Du), len(keep))
        return True, removed
    return False, 0


def ac3(
    problem,
    domains: Optional[Dict[Variable, List[Value]]] = None,
    *,
    in_place: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[Variable, List[Value]]]:
    """
    Enforce arc consistency using AC-3 over binary constraints.

    Parameters
    ----------
    problem : object
        Must expose `variables`, `domains`, `constraints`, and optionally `neighbors(var)`.
    domains : dict | None
        Custom domains to filter; defaults to `problem.domains`.
    in_place : bool, default False
        If True, mutate the given `domains`. Otherwise, a deep copy is filtered.
    logger : logging.Logger | None
        Optional logger override (defaults to module logger).

    Returns
    -------
    Dict[Variable, List[Value]] | None
        Filtered domains, or None if a domain wipeout occurs (UNSAT).

    Notes
    -----
    - Complexity is O(e * d^2) in the worst case (e=number of arcs, d=max domain size).
    """
    logx = logger or log

    # Copy domains unless we’re asked to mutate in-place
    base = problem.domains if domains is None else domains
    D: Dict[Variable, List[Value]]
    if in_place:
        D = base
    else:
        D = {v: list(vals) for v, vals in base.items()}

    # Precompute constraints by arc to avoid re-scanning on every REVISE
    cons_map: Dict[Tuple[Variable, Variable], List[BinaryConstraint]] = {}
    for c in problem.constraints:
        cons_map.setdefault((c.u, c.v), []).append(c)
        cons_map.setdefault((c.v, c.u), []).append(c)  # allow reverse lookup too

    # Build initial queue of all directed arcs (Xi, Xj) that actually have constraints
    Q: deque[Tuple[Variable, Variable]] = deque()
    for v in problem.variables:
        for nb in _neighbors_of(problem, v):
            if (v, nb) in cons_map:
                Q.append((v, nb))

    total_removed = 0
    logx.info("AC3: start with %d arcs", len(Q))

    while Q:
        u, v = Q.popleft()
        if not D[u] or not D[v]:
            logx.debug("AC3: early stop, wipeout on %s or %s", u, v)
            return None
        changed, removed = _revise(problem, u, v, D, cons_map[(u, v)])
        total_removed += removed
        if changed:
            if not D[u]:
                logx.info("AC3: domain wipeout on %s", u)
                return None
            # If Du changed, all arcs (w, u) for w ≠ v must be rechecked
            for w in _neighbors_of(problem, u):
                if w != v and (w, u) in cons_map:
                    Q.append((w, u))

    logx.info("AC3: done; total removed=%d", total_removed)
    return D
