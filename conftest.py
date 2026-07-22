"""
conftest.py — root-level pytest configuration.

Inserts src/ onto sys.path so all tests can import modules by their package
name (e.g. `from config import get_config`, `from data.preprocessing import ...`)
without per-file sys.path hacks.

This replaces the per-file `sys.path.insert(0, ...)` pattern and matches
the convention used by scripts/ (see AGENTS.md).
"""

import sys
from pathlib import Path

# Make sure src/ is on the path before any test module is collected
SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
