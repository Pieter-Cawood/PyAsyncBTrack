# src/pyasyncbtrack/solver.py
from __future__ import annotations

"""
Recursive ABT-style solver (centralized) with heuristics, logging, multi-solution support,
AC-3 pre-pruning, restarts, and selectable progress verbosity.

Highlights
----------
- MRV + degree tie-break (var selection), LCV (value order), forward checking
- Conflict-directed backjumping (deepest culprit)
- Iteration-cap restarts with optional domain reshuffling
- Optional AC-3 domain prefilter (per run)
- Multi-solution enumeration with *provably unique* results
- Verbosity as Enum: OFF / LOG / TQDM (desc-only line)
- Per-run diversified RNG (SplitMix64-style) for widely different reshuffles

Kwargs
------
timeout_s: float | None               (default 10.0)  Global wall-clock budget.
use_mrv: bool                         (default True)  MRV heuristic.
use_lcv: bool                         (default True)  LCV heuristic.
domain_reshuffling: bool              (default True)  Shuffle domains per run.
random_tiebreak: bool                 (default True)  Tie-breaking jitter.
rng: random.Random | None             (default None)  External RNG instance.
seed: int | None                      (default None)  Seed (ignored if rng given).
reshuffle_iterations: int | None      (default None)  Iteration cap per run.
prefilter_domain: bool                (default False) AC-3 before each run.
verbosity: Verbosity                  (default OFF)   OFF/LOG/TQDM progress.
nr_of_solutions: int | None           (default None)  Target # of distinct sols.
solutions_timeout_s: float | None     (default None)  Time budget for enumeration.
progress_log_every: int               (default 5000)  LOG update cadence (iterations).
diversify_restarts: bool              (default True)  New per-run RNG for each restart.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple, Union, Set, Any, Callable
from enum import Enum, auto
import time
import random
import logging

try:
    from tqdm.auto import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

from .types import Variable, Value, Assignment
from .agent import Agent
from .consistency import ac3

log = logging.getLogger("pyasyncbtrack.solver")


# =============================================================================
# Verbosity
# =============================================================================

class Verbosity(Enum):
    OFF = auto()
    LOG = auto()
    TQDM = auto()


# =============================================================================
# Config (kwargs)
# =============================================================================

_DEFAULTS: Dict[str, Any] = {
    "timeout_s": 10.0,
    "use_mrv": True,
    "use_lcv": True,
    "domain_reshuffling": True,
    "random_tiebreak": True,
    "rng": None,
    "seed": None,
    "reshuffle_iterations": None,
    "prefilter_domain": False,
    "verbosity": Verbosity.OFF,
    "nr_of_solutions": None,
    "solutions_timeout_s": None,
    "progress_log_every": 5000,
    "diversify_restarts": True,
}


# =============================================================================
# Internals
# =============================================================================

@dataclass
class _NogoodReturn:
    culprit: Variable
    nogood: Dict[Variable, Value]


def _neighbors_of(problem, var: Variable) -> List[Variable]:
    """Get neighbors(var) from problem if available; else derive from constraints."""
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


# --- Canonicalization for guaranteed-unique solutions ------------------------

def _canon_val(v: Value) -> Any:
    """Canonical, hashable representation for Value (handles tuples/frozensets recursively)."""
    if isinstance(v, tuple):
        return ("T", tuple(_canon_val(x) for x in v))
    if isinstance(v, frozenset):
        return ("F", tuple(sorted((_canon_val(x) for x in v), key=repr)))
    return ("S", v)


def _make_solution_key(assignment: Assignment, var_order: Tuple[Variable, ...]) -> Tuple[Tuple[Variable, Any], ...]:
    """Build a canonical, hashable fingerprint of a full assignment."""
    return tuple((v, _canon_val(assignment[v])) for v in var_order)


# --- Per-run RNG diversification (SplitMix64-style) --------------------------

_MASK64 = (1 << 64) - 1

def _mix64(x: int) -> int:
    """SplitMix64 mixer for well-spread 64-bit seeds."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK64
    z ^= (z >> 31)
    return z & _MASK64

def _derive_run_rng(base_rng: random.Random, run_index: int) -> Tuple[random.Random, int]:
    """
    Derive a per-run RNG that's far apart in state from prior runs,
    ensuring reshuffles explore very different regions (but remain
    reproducible for the same base seed).
    Returns (rng, seed64).
    """
    base_bits = base_rng.getrandbits(64)
    seed64 = _mix64(base_bits ^ ((run_index & _MASK64) * 0x9E3779B97F4A7C15 & _MASK64))
    rr = random.Random(seed64)
    return rr, seed64


# --- Progress (TQDM desc-only or periodic LOG) -------------------------------

_TQDM_FORMAT = "{desc}"  # no bar/ETA/rate


class _Progress:
    """Per-run progress tracker with selectable backend and extra fields (sols/time%)."""
    def __init__(
        self,
        verbosity: Verbosity,
        log_every: int = 5000,
        sols_text_fn: Optional[Callable[[], str]] = None,
        time_pct_fn: Optional[Callable[[], Optional[float]]] = None,
    ) -> None:
        self.verbosity = verbosity
        self.log_every = max(1, int(log_every))
        self.n = 0
        self.reshuffles = 0
        self.max_for_run: Optional[int] = None
        self._bar = None
        self._last_logged = 0
        self._sols_text_fn = sols_text_fn
        self._time_pct_fn = time_pct_fn

        if self.verbosity == Verbosity.TQDM:
            if _HAS_TQDM:
                self._bar = tqdm(total=0, desc="iter 0/∞ • reshuffles=0", bar_format=_TQDM_FORMAT, ncols=0, leave=False)
            else:
                log.warning("Verbosity.TQDM requested but tqdm is not available; falling back to LOG")
                self.verbosity = Verbosity.LOG

    def _desc(self) -> str:
        parts = [f"iter {self.n}/{self.max_for_run if self.max_for_run is not None else '∞'}",
                 f"reshuffles={self.reshuffles}"]
        if self._sols_text_fn is not None:
            try:
                parts.insert(1, f"sols {self._sols_text_fn()}")
            except Exception:
                pass
        if self._time_pct_fn is not None:
            try:
                p = self._time_pct_fn()
                if p is not None:
                    pct = int(max(0.0, min(1.0, p)) * 100.0)
                    parts.append(f"time={pct}%")
            except Exception:
                pass
        return " • ".join(parts)

    def reset_run(self, max_for_run: Optional[int]) -> None:
        self.n = 0
        self.max_for_run = max_for_run
        self._last_logged = 0
        if self.verbosity == Verbosity.TQDM and self._bar is not None:
            total = max_for_run if max_for_run is not None else 0
            try:
                self._bar.reset(total=total)  # type: ignore[attr-defined]
            except Exception:
                self._bar.close()
                self._bar = tqdm(total=total, desc="", bar_format=_TQDM_FORMAT, ncols=0, leave=False)
            self._bar.set_description(self._desc())
            self._bar.refresh()
        elif self.verbosity == Verbosity.LOG:
            log.info(self._desc())

    def bump_reshuffles(self) -> None:
        self.reshuffles += 1

    def step(self, k: int = 1) -> None:
        self.n += k
        if self.verbosity == Verbosity.TQDM and self._bar is not None:
            self._bar.update(k)
            self._bar.set_description(self._desc())
        elif self.verbosity == Verbosity.LOG:
            if (self.n - self._last_logged) >= self.log_every:
                self._last_logged = self.n
                log.info(self._desc())

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


# =============================================================================
# Public API
# =============================================================================

class _RestartRun(RuntimeError):
    """Abort a run when the per-run iteration cap is reached."""
    pass


def solve(problem, **kwargs) -> Union[Assignment, List[Assignment], None]:
    """
    Solve a binary DCSP using a centralized, recursive ABT-style search.

    See module docstring for kwargs and defaults. Only pass what you need:
    >>> solve(problem, timeout_s=30.0, nr_of_solutions=5, verbosity=Verbosity.LOG)
    """
    # Merge & unpack config
    cfg = {**_DEFAULTS, **kwargs}

    timeout_s: Optional[float]       = cfg["timeout_s"]
    use_mrv: bool                    = cfg["use_mrv"]
    use_lcv: bool                    = cfg["use_lcv"]
    domain_reshuffling: bool         = cfg["domain_reshuffling"]
    random_tiebreak: bool            = cfg["random_tiebreak"]
    rng: Optional[random.Random]     = cfg["rng"]
    seed: Optional[int]              = cfg["seed"]
    reshuffle_iterations: Optional[int] = cfg["reshuffle_iterations"]
    prefilter_domain: bool           = cfg["prefilter_domain"]
    verbosity: Verbosity             = cfg["verbosity"]
    nr_of_solutions: Optional[int]   = cfg["nr_of_solutions"]
    solutions_timeout_s: Optional[float] = cfg["solutions_timeout_s"]
    progress_log_every: int          = cfg["progress_log_every"]
    diversify_restarts: bool         = cfg["diversify_restarts"]

    # RNG
    if rng is not None:
        R = rng
        log.info("solve: using provided RNG (seed ignored)")
    else:
        R = random.Random(seed)
        if seed is not None:
            log.info("solve: using seed=%s", seed)

    cap = None if not reshuffle_iterations or reshuffle_iterations <= 0 else int(reshuffle_iterations)
    target_k = None if (nr_of_solutions is None or nr_of_solutions <= 1) else int(nr_of_solutions)

    # Enumeration mode?
    enumerate_mode = (target_k is not None and target_k >= 2) or (solutions_timeout_s is not None and solutions_timeout_s > 0)

    # Fixed var order for canonical keys
    var_order: Tuple[Variable, ...] = tuple(sorted(list(problem.variables)))

    # Timers
    start_wall = time.time()
    start_enum = time.time() if enumerate_mode and solutions_timeout_s and solutions_timeout_s > 0 else None

    def remaining_global() -> Optional[float]:
        if timeout_s is None or timeout_s <= 0:
            return None
        return max(0.0, timeout_s - (time.time() - start_wall))

    def remaining_enum() -> Optional[float]:
        if start_enum is None or solutions_timeout_s is None or solutions_timeout_s <= 0:
            return None
        return max(0.0, solutions_timeout_s - (time.time() - start_enum))

    def remaining_overall() -> Optional[float]:
        rg = remaining_global()
        re = remaining_enum()
        candidates = [t for t in (rg, re) if t is not None]
        return min(candidates) if candidates else None

    # time% should only reflect multi-solution timeout (if enabled)
    def time_pct() -> Optional[float]:
        if start_enum is None or solutions_timeout_s is None or solutions_timeout_s <= 0:
            return None
        elapsed = time.time() - start_enum
        return max(0.0, min(1.0, elapsed / solutions_timeout_s))

    # Global, canonical set of seen solutions (guarantees uniqueness)
    seen_keys: Set[Tuple[Tuple[Variable, Any], ...]] = set()

    # Human-readable store of solutions
    all_solutions: List[Assignment] = []

    # sols text
    def sols_text_single() -> str:
        return f"{len(all_solutions)}/1"

    def sols_text_enum() -> str:
        total = str(target_k) if target_k is not None else "∞"
        return f"{len(all_solutions)}/{total}"

    # Helper: get one (new) solution honoring budgets and restart cap
    def _one_solution(progress: _Progress) -> Optional[Assignment]:
        try:
            while True:
                left = remaining_overall()
                if left is not None and left <= 0:
                    log.info("solve: time budget exhausted; reshuffles=%d", progress.reshuffles)
                    return None

                progress.bump_reshuffles()
                progress.reset_run(max_for_run=cap)

                # Derive a per-run RNG to widely diversify the search
                if diversify_restarts:
                    run_rng, run_seed = _derive_run_rng(R, progress.reshuffles)
                    log.info("solve: start run #%d (cap=%s, prefilter_domain=%s, run_seed=%d)",
                             progress.reshuffles, str(cap), prefilter_domain, run_seed)
                else:
                    run_rng = R
                    log.info("solve: start run #%d (cap=%s, prefilter_domain=%s)", progress.reshuffles, str(cap), prefilter_domain)

                try:
                    res = _solve_once(
                        problem=problem,
                        timeout_s=left,
                        use_mrv=use_mrv,
                        use_lcv=use_lcv,
                        domain_reshuffling=domain_reshuffling,
                        random_tiebreak=random_tiebreak,
                        rng=run_rng,  # per-run RNG
                        progress=progress,
                        iter_cap_for_run=cap,
                        prefilter_domain=prefilter_domain,
                        seen_keys=seen_keys,      # uniqueness gate
                        var_order=var_order,
                    )
                except _RestartRun:
                    log.info("solve: run #%d hit iteration cap (%s) → reshuffle & retry", progress.reshuffles, str(cap))
                    continue

                if isinstance(res, dict):
                    return res  # already registered into seen_keys inside run

                log.info("solve: run #%d ended without solution (iters=%d); continuing until timeout",
                         progress.reshuffles, progress.n)
        finally:
            progress.close()

    # Modes
    if not enumerate_mode:
        progress = _Progress(
            verbosity=verbosity,
            log_every=progress_log_every,
            sols_text_fn=sols_text_single,
            time_pct_fn=None,  # no time% in single-solution mode
        )
        sol = _one_solution(progress)
        return sol

    # Enumeration loop
    while True:
        left = remaining_overall()
        if left is not None and left <= 0:
            break

        progress = _Progress(
            verbosity=verbosity,
            log_every=progress_log_every,
            sols_text_fn=sols_text_enum,
            time_pct_fn=(time_pct if (solutions_timeout_s is not None and solutions_timeout_s > 0) else None),
        )
        sol = _one_solution(progress)
        if sol is None:
            break

        all_solutions.append(sol)
        log.info("solve: collected %d%s distinct solutions",
                 len(all_solutions),
                 f"/{target_k}" if target_k is not None else "")

        if target_k is not None and len(all_solutions) >= target_k:
            break

    return all_solutions


# Backwards-compat alias
solve_recursive = solve


# =============================================================================
# Single run
# =============================================================================

class _RestartRun(RuntimeError):
    """Abort a run when the per-run iteration cap is reached."""
    pass


def _solve_once(
    problem,
    *,
    timeout_s: Optional[float],
    use_mrv: bool,
    use_lcv: bool,
    domain_reshuffling: bool,
    random_tiebreak: bool,
    rng: random.Random,
    progress: _Progress,
    iter_cap_for_run: Optional[int],
    prefilter_domain: bool,
    seen_keys: Set[Tuple[Tuple[Variable, Any], ...]],  # canonical set shared across runs
    var_order: Tuple[Variable, ...],
) -> Optional[Assignment]:
    """
    Execute a single run. Raises `_RestartRun` when `iter_cap_for_run` is exceeded.
    Returns a full assignment on success (and *registers it in seen_keys*), or None.
    """
    start = time.time()

    def timed_out() -> bool:
        return timeout_s is not None and timeout_s > 0 and (time.time() - start) > timeout_s

    # Build working domains for this run
    run_domains: Dict[Variable, List[Value]] = {v: list(problem.domains[v]) for v in problem.variables}

    # Optional AC-3 pre-pass
    if prefilter_domain:
        filtered = ac3(problem, domains=run_domains, in_place=False, logger=log)
        if filtered is None:
            log.info("run: AC-3 wiped out a domain → UNSAT for this run")
            return None
        run_domains = filtered
        log.info("run: AC-3 applied (domains possibly pruned)")

    # Optional randomization AFTER AC-3 so order bias is reduced on pruned sets
    if domain_reshuffling:
        for v in run_domains:
            rng.shuffle(run_domains[v])

    # Build agents (fresh each run)
    agents: Dict[Variable, Agent] = {}
    for v in problem.variables:
        agents[v] = Agent(
            var=v,
            domain=list(run_domains[v]),
            neighbors_all=_neighbors_of(problem, v),
            constraints=list(problem.constraints),
        )

    # Global assignment/state for this run
    assign: Assignment = {}
    trail: List[Variable] = []            # assignment order
    depth_index: Dict[Variable, int] = {} # var -> depth (for deepest-culprit)

    def push_assignment(var: Variable, val: Value) -> None:
        """Assign `var=val`, publish to neighbor views, update depth & progress."""
        progress.step(1)
        if iter_cap_for_run is not None and progress.n > iter_cap_for_run:
            raise _RestartRun()

        a = agents[var]
        a.value = val
        assign[var] = val
        trail.append(var)
        depth_index[var] = len(trail) - 1
        for nb in a.neighbors_all:
            agents[nb].view[var] = val

    def pop_assignment(var: Variable) -> None:
        """Undo assignment of `var` and retract from neighbor views."""
        a = agents[var]
        for nb in a.neighbors_all:
            agents[nb].view.pop(var, None)
        assign.pop(var, None)
        a.value = None
        if trail and trail[-1] == var:
            trail.pop()
        else:
            try:
                trail.remove(var)
            except ValueError:
                pass
        depth_index.pop(var, None)

    def pick_culprit(ng: Dict[Variable, Value]) -> Variable:
        """Choose deepest assigned variable from the nogood (conflict-directed backjumping)."""
        assigned = [v for v in ng.keys() if v in depth_index]
        if not assigned:
            keys = list(ng.keys())
            if random_tiebreak and len(keys) > 1:
                rng.shuffle(keys)
            return keys[-1]
        return max(assigned, key=lambda v: depth_index[v])

    def learn(agent: Agent, ng: Dict[Variable, Value]) -> None:
        """Learn a nogood with light subsumption (drop weaker/duplicate)."""
        new_items = frozenset(ng.items())
        for existing in agent.nogoods:
            if frozenset(existing.items()) <= new_items:
                return
        filtered: List[Dict[Variable, Value]] = []
        for existing in agent.nogoods:
            if not (new_items <= frozenset(existing.items())):
                filtered.append(existing)
        agent.nogoods = filtered
        agent.nogoods.append(dict(ng))

    def forward_check_on(unassigned: Iterable[Variable]) -> Optional[_NogoodReturn]:
        """Detect immediate wipeouts in unassigned neighbors; return view-nogood if any."""
        for n in unassigned:
            a = agents[n]
            eff = a.effective_domain()
            if not eff:
                ng = dict(a.view)
                if not ng:
                    return None  # global unsat at top of run (rare)
                return _NogoodReturn(culprit=pick_culprit(ng), nogood=ng)
        return None

    def select_var(unassigned: List[Variable]) -> Variable:
        """MRV + degree tie-break + optional random tiebreak."""
        if not use_mrv:
            return unassigned[0]
        scored: List[Tuple[int, int, int, Variable]] = []
        for v in unassigned:
            a = agents[v]
            mrv = len(a.effective_domain())
            deg = sum(1 for nb in a.neighbors_all if nb in unassigned and nb != v)
            rb = rng.randrange(1 << 30) if random_tiebreak else 0
            scored.append((mrv, -deg, rb, v))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        return scored[0][3]

    def order_values(var: Variable, candidates: List[Value], unassigned: List[Variable]) -> List[Value]:
        """LCV value ordering (favor values that leave more neighbor supports)."""
        if not use_lcv or len(unassigned) == 0 or len(candidates) <= 1:
            return candidates
        a = agents[var]
        scores: List[Tuple[int, int, Value]] = []
        for val in candidates:
            total_support = 0
            for nb in a.neighbors_all:
                if nb not in unassigned or nb == var:
                    continue
                nb_a = agents[nb]
                supports_nb = 0
                for cand_nb in nb_a.domain:
                    # Respect nb's nogoods under its current view
                    blocked = False
                    for ng in nb_a.nogoods:
                        others_match = all(nb_a.view.get(k) == vv for k, vv in ng.items() if k != nb)
                        if others_match and nb in ng and ng[nb] == cand_nb:
                            blocked = True
                            break
                        if others_match and (nb not in ng):
                            blocked = True
                            break
                    if blocked:
                        continue
                    # Consistency vs nb's view and (var=val)
                    ok = True
                    for hv, hvv in nb_a.view.items():
                        if not nb_a.consistent_pair(cand_nb, hv, hvv):
                            ok = False
                            break
                    if ok and not nb_a.consistent_pair(cand_nb, var, val):
                        ok = False
                    if ok:
                        supports_nb += 1
                total_support += supports_nb
            jitter = rng.randrange(4) if random_tiebreak else 0
            scores.append((-total_support, jitter, val))  # more supports → better (smaller negative)
        scores.sort(key=lambda t: (t[0], t[1]))
        return [v for _, __, v in scores]

    def extend(unassigned: List[Variable]) -> Optional[Assignment] | _NogoodReturn:
        if timed_out():
            return None
        if not unassigned:
            # Canonical duplicate gate: register or reject
            key = _make_solution_key(assign, var_order)
            if key in seen_keys:
                ng = dict(assign)  # full-assignment nogood
                culprit = pick_culprit(ng)
                return _NogoodReturn(culprit=culprit, nogood=ng)
            # Register immediately (so outer collection cannot see duplicates)
            seen_keys.add(key)
            # Validate (paranoia)
            for c in problem.constraints:
                if not c.predicate(c.u, assign[c.u], c.v, assign[c.v]):
                    return None
            return dict(assign)

        var = select_var(unassigned)
        a = agents[var]

        eff = a.effective_domain()
        if not eff:
            ng = dict(a.view)
            if not ng:
                return None
            return _NogoodReturn(culprit=pick_culprit(ng), nogood=ng)

        vals = order_values(var, eff, unassigned)

        for v in vals:
            push_assignment(var, v)

            rest = [x for x in unassigned if x != var]
            fc = forward_check_on(rest)
            if isinstance(fc, _NogoodReturn):
                learn(a, fc.nogood)
                if var in fc.nogood:
                    pass  # try next value
                else:
                    pop_assignment(var)
                    return fc
            else:
                res = extend(rest)
                if isinstance(res, dict):
                    return res
                if isinstance(res, _NogoodReturn):
                    learn(a, res.nogood)
                    if var in res.nogood:
                        pass
                    else:
                        pop_assignment(var)
                        return res

            pop_assignment(var)

        # No value works for this var → return higher-only nogood (the view)
        ng = dict(a.view)
        if not ng:
            return None
        learn(a, ng)
        culprit = pick_culprit(ng)
        a.view.pop(culprit, None)  # force reconsideration upstream
        return _NogoodReturn(culprit=culprit, nogood=ng)

    # Execute one run (short-circuit via _RestartRun when per-run cap is hit)
    unassigned0 = list(problem.variables)
    if random_tiebreak:
        rng.shuffle(unassigned0)
    res = extend(unassigned0)
    if isinstance(res, dict):
        return res
    return None
