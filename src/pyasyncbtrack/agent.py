# src/pyasyncbtrack/agent.py
from __future__ import annotations

"""
Agent (centralized/recursive ABT helper).

This module defines a lightweight `Agent` data structure used by the
centralized, recursive ABT-style solver. Each agent owns:

- a single decision variable (`var`)
- its local domain (`domain`)
- a view of earlier-assigned neighbors (`view`)
- locally learned nogoods (`nogoods`)

The solver orchestrates assignment order, backjumping, and messaging
semantics. The `Agent` class provides fast, pure helpers for:
- pairwise consistency checks against binary constraints
- filtering a domain with respect to the current view and nogoods
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .types import BinaryConstraint, Variable, Value


@dataclass
class Agent:
    """
    ABT agent state for centralized/recursive solving.

    Parameters
    ----------
    var : Variable
        Identifier of the variable owned by this agent.
    domain : List[Value]
        Domain values (may be pre-shuffled by the solver).
    neighbors_all : List[Variable]
        All binary neighbors of `var`.
    constraints : List[BinaryConstraint]
        Global list of binary constraints (the agent filters to relevant ones).

    Attributes
    ----------
    view : Dict[Variable, Value]
        Latest known assignments of already-processed neighbors.
    value : Optional[Value]
        The agent's current assignment (None if unassigned).
    nogoods : List[Dict[Variable, Value]]
        Locally learned conflict sets (higher-only or including `var`).
    """
    var: Variable
    domain: List[Value]
    neighbors_all: List[Variable]
    constraints: List[BinaryConstraint]

    view: Dict[Variable, Value] = field(default_factory=dict)
    value: Optional[Value] = None
    nogoods: List[Dict[Variable, Value]] = field(default_factory=list)

    # --------------------------------------------------------------------- #
    # Consistency helpers (pure, fast)                                      #
    # --------------------------------------------------------------------- #

    def consistent_pair(self, v: Value, other_var: Variable, other_val: Value) -> bool:
        """
        Check pairwise consistency between (self.var=v) and (other_var=other_val).

        Returns
        -------
        bool
            True iff all constraints between `self.var` and `other_var` are satisfied.
        """
        for c in self.constraints:
            if ((c.u == self.var and c.v == other_var) or
                (c.u == other_var and c.v == self.var)):
                u_val = v if c.u == self.var else other_val
                v_val = other_val if c.v == other_var else v
                if not c.predicate(c.u, u_val, c.v, v_val):
                    return False
        return True

    def candidate_blocked_by_nogood(self, v: Value) -> bool:
        """
        Test whether value `v` is forbidden by any active nogood.

        A nogood blocks `v` if:
        - all of its *other* bindings match `self.view`, and
        - it either binds (self.var == v), or is a higher-only nogood (no self.var),
          which means the current view is inconsistent regardless of our value.
        """
        for ng in self.nogoods:
            others_match = all(self.view.get(k) == val for k, val in ng.items() if k != self.var)
            if not others_match:
                continue
            if self.var in ng and ng[self.var] == v:
                return True
            if self.var not in ng:
                return True
        return False

    def effective_domain(self) -> List[Value]:
        """
        Filter `domain` by nogoods and pairwise consistency with `view`.

        Returns
        -------
        List[Value]
            Values that are not excluded by nogoods and are consistent with
            the current neighbor view.
        """
        eff: List[Value] = []
        for v in self.domain:
            if self.candidate_blocked_by_nogood(v):
                continue
            ok = True
            for hv, hvv in self.view.items():
                if not self.consistent_pair(v, hv, hvv):
                    ok = False
                    break
            if ok:
                eff.append(v)
        return eff
