
def test_import_and_solve():
    from pyasyncbtrack import DCSPProblem, solve, Verbosity
    from pyasyncbtrack.constraints import not_equal

    variables = ["X", "Y"]
    domains = { "X": [1, 2], "Y": [1, 2] }
    constraints = [not_equal("X", "Y")]
    problem = DCSPProblem(variables, domains, constraints)

    sol = solve(problem, verbosity=Verbosity.OFF, seed=1)
    assert sol is not None
    assert sol["X"] != sol["Y"]
