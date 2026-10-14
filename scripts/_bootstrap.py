"""
_bootstrap.py — import this FIRST in every script, before any src/ import.

Sets HF cache defaults (so nothing lands in ~/.cache on the small root disk)
and puts the repo root on sys.path. env.sh / .env values always win because
we only setdefault.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# HF cache: env.sh/.env wins; otherwise a repo-local cache (repo sits on the
# big scratch mount on the server, so this is safe by construction).
os.environ.setdefault("HF_HOME", str(REPO / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
# Headless figures; harmless if already set.
os.environ.setdefault("MPLBACKEND", "Agg")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
