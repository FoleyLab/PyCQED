#!/usr/bin/env python3
"""Run FSSH trajectories from a TOML config: ``python scripts/run_fssh.py examples/legacy_double_well.toml``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pycqed.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
