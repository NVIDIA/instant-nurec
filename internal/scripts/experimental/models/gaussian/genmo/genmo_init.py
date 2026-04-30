"""Initialize GenMO PROJ_ROOT for Bazel runfiles environment.

This module MUST be imported before any GenMO/hmr4d modules to ensure
the correct project root is set for finding checkpoints and data files.

Usage:
    from internal.scripts.experimental.models.gaussian.genmo import genmo_init
    # Now safe to import GenMO modules
    import hmr4d
"""

import os

from pathlib import Path

from python.runfiles import runfiles


def init_genmo_proj_root():
    """Initialize GenMO PROJ_ROOT from Bazel runfiles.

    Returns:
        Path: The GenMO project root directory

    Raises:
        FileNotFoundError: If GenMO inputs cannot be found in runfiles
    """
    runfiles_obj = runfiles.Create()
    repo_path = runfiles_obj.Rlocation("+_repo_rules5+genmo_inputs/inputs/nvhuman_data/nvhuman_lite.npz")

    if repo_path and os.path.exists(repo_path):
        genmo_proj_root = Path(repo_path).parent.parent.parent
        # Set environment variable for any GenMO code that checks it
        os.environ["GENMO_PROJ_ROOT"] = str(genmo_proj_root)
        return genmo_proj_root
    else:
        raise FileNotFoundError(
            "Could not find GenMO inputs in Bazel runfiles. "
            "Tried: +_repo_rules5+genmo_inputs/inputs/nvhuman_data/nvhuman_lite.npz"
        )


# Initialize immediately when this module is imported
_GENMO_PROJ_ROOT = init_genmo_proj_root()

# Import hmr4d and set PROJ_ROOT
import hmr4d


hmr4d.PROJ_ROOT = _GENMO_PROJ_ROOT


def get_genmo_proj_root() -> Path:
    """Get the initialized GenMO project root.

    Returns:
        Path: The GenMO project root directory
    """
    return _GENMO_PROJ_ROOT
