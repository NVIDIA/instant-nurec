When you work on this code base, you must adhere to the following rules:

1. Branch-coverage tests on every function (TDD).
2. Run tests with `.venv/bin/python -m pytest tests/ -q`.
3. Lint with `.venv/bin/ruff check .`.
4. End-to-end inference and parity-style verification require a GPU; only those calls may run unsandboxed.
5. One logical change per commit. Subject line: `<type>(<area>): <imperative one-liner>`.
