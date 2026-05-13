When you work on this code base, you must adhere to the following rules:

1. Branch-coverage tests on every function (TDD).
2. Run tests with `.venv/bin/python -m pytest tests/ -q`.
3. Lint with `.venv/bin/ruff check .`.
4. End-to-end inference and parity-style verification require a GPU; only those calls may run unsandboxed.
5. Use `git commit --fixup=<SHA>` commits to fix old commits, don't create new commits. Ammend if the fixup is for the last commit.
6. If it's not a fixup or amend, do one logical change per commit. Subject line: `<type>(<area>): <imperative one-liner>`.
7. Before each commit, ensure parity, see 4. Then state the delta in number of vertices for merge and no-merge cases wrt the baseline. Also state the runtimes for merge and no-merge. Also state Chamfer and F1@0.01 metrics. If you don't know the baseline, ask for it.
