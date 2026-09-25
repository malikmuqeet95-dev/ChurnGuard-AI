"""Pytest bootstrap for resolving imports from backend/src.

The application code lives under ``backend/src``, but the tests import modules
as ``src.*``. Adding ``backend`` to ``sys.path`` here keeps the test imports
stable without forcing a package rename or per-test path hacks.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
