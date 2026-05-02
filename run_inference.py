#!/usr/bin/env python3
"""Standalone Kelvin predict entrypoint.

Invocation::

    python run_inference.py --ncore-path <path> --output-dir <path> --merge {none,frustum-ownership}

Phase 3 step 8.4: this is the canonical invocation. The Phase 1 ``bazel run
//instant_nurec:run -- ...`` form remains supported through Phase 3
transition; ``run_inference.py`` defers to ``instant_nurec.cli.main`` and so
runs whichever tree is currently importable on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec.cli import main


if __name__ == "__main__":
    sys.exit(main())
