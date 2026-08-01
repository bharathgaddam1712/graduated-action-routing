"""
session_init.py — Run this as the FIRST cell of every Kaggle notebook.

What it does:
1. Detects whether running on Kaggle or locally
2. On Kaggle: copies src/ + results/ from persistent input datasets into /kaggle/working/
3. Sets HuggingFace cache to the pre-cached model dataset (avoids re-downloading ~20GB)
4. Returns a Paths object with all key directories

Three Kaggle Datasets needed:
  slm-agents-code    → contains src/ folder
  slm-agents-results → contains results/ CSVs from prior sessions
  slm-agents-models  → contains hf_cache/ with downloaded model weights (created in step 0)
"""

import os
import sys
import shutil
from pathlib import Path


def init(base_dir: str | None = None) -> dict[str, Path]:
    """
    Initialize the session. Returns a dict of key paths.
    Call at the top of every notebook:
        from src.session_init import init
        P = init()
    """
    on_kaggle = Path("/kaggle/working").exists()

    if on_kaggle:
        WORK = Path("/kaggle/working/slm_agents_paper")
    else:
        WORK = Path(base_dir) if base_dir else Path(__file__).parent.parent

    SRC     = WORK / "src"
    DATA    = WORK / "data"
    RESULTS = WORK / "results"
    FIGURES = WORK / "figures"

    for d in [SRC, DATA, RESULTS, FIGURES]:
        d.mkdir(parents=True, exist_ok=True)

    if on_kaggle:
        _sync_from_kaggle_datasets(WORK, SRC, RESULTS)

    # Add src to sys.path
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))

    paths = {
        "work": WORK,
        "src": SRC,
        "data": DATA,
        "results": RESULTS,
        "figures": FIGURES,
    }

    print("=== Session initialized ===")
    for k, v in paths.items():
        print(f"  {k:10s}: {v}")
    print()
    return paths


def _sync_from_kaggle_datasets(work: Path, src: Path, results: Path) -> None:
    """Copy src/ and results/ from Kaggle input datasets into working directory."""

    # ── 1. Sync src/ from slm-agents-code ─────────────────────────────────
    code_input = Path("/kaggle/input/slm-agents-code")
    if code_input.exists():
        src_input = code_input / "src"
        if src_input.exists():
            shutil.copytree(src_input, src, dirs_exist_ok=True)
            print(f"  [init] Synced src/ from {code_input}")
        else:
            print(f"  [init] WARNING: {src_input} not found in slm-agents-code dataset")
    else:
        print("  [init] slm-agents-code dataset not attached — using notebook's own src/")

    # ── 2. Sync results/ from slm-agents-results ──────────────────────────
    results_input = Path("/kaggle/input/slm-agents-results")
    if results_input.exists():
        for f in results_input.glob("*.csv"):
            dest = results / f.name
            if not dest.exists():          # don't overwrite results written this session
                shutil.copy2(f, dest)
                print(f"  [init] Restored {f.name} from slm-agents-results")
        # Also restore JSON files (bin_data, logit features, etc.)
        for f in results_input.glob("*.json"):
            dest = results / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                print(f"  [init] Restored {f.name} from slm-agents-results")
    else:
        print("  [init] No prior results found (first session or dataset not attached)")

    # ── 3. Set HuggingFace cache to pre-downloaded models ─────────────────
    model_cache = Path("/kaggle/input/slm-agents-models/hf_cache")
    if model_cache.exists():
        os.environ["HF_HOME"]               = str(model_cache)
        os.environ["TRANSFORMERS_CACHE"]    = str(model_cache / "hub")
        os.environ["HF_DATASETS_CACHE"]     = str(model_cache / "datasets")
        print(f"  [init] HF cache → {model_cache}")
    else:
        # Fall back to a writable working dir cache
        fallback = Path("/kaggle/working/hf_cache")
        fallback.mkdir(exist_ok=True)
        os.environ["HF_HOME"]            = str(fallback)
        os.environ["TRANSFORMERS_CACHE"] = str(fallback / "hub")
        os.environ["HF_DATASETS_CACHE"]  = str(fallback / "datasets")
        print(f"  [init] slm-agents-models not attached — HF cache → {fallback}")
        print(f"  [init] Models will be downloaded fresh this session")
