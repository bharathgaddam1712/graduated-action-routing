"""
session_save.py — Run this as the LAST cell of every Kaggle notebook.

What it does:
1. Zips all results/ files into /kaggle/working/results_snapshot.zip
2. Prints step-by-step instructions to persist results before closing the session

You MUST download results_snapshot.zip and re-upload it as a new version of the
`slm-agents-results` Kaggle Dataset before closing the session, or progress is lost.
"""

import zipfile
import shutil
from pathlib import Path
from datetime import datetime


def save(results_dir: str | Path | None = None) -> Path:
    """
    Zip all results and print upload instructions.
    Returns path to the zip file.
    """
    on_kaggle = Path("/kaggle/working").exists()

    if results_dir is None:
        results_dir = Path("/kaggle/working/slm_agents_paper/results") if on_kaggle \
                      else Path(__file__).parent.parent / "results"

    results_dir = Path(results_dir)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M")
    zip_path    = Path("/kaggle/working") / f"results_snapshot_{timestamp}.zip" if on_kaggle \
                  else results_dir.parent / f"results_snapshot_{timestamp}.zip"

    # Create zip
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(results_dir.glob("*")):
            if f.is_file():
                zf.write(f, f.name)
                print(f"  [save] Zipped: {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    print(f"\n✅ Results zipped → {zip_path}")
    print(f"   Size: {zip_path.stat().st_size / 1024:.1f} KB")

    if on_kaggle:
        _print_upload_instructions(zip_path, timestamp)

    return zip_path


def _print_upload_instructions(zip_path: Path, timestamp: str) -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  IMPORTANT — Save your progress before closing this session          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Step 1: Click the output icon in the right Kaggle sidebar           ║
║  Step 2: Download:  results_snapshot_{ts}.zip                        ║
║  Step 3: Unzip it locally                                            ║
║  Step 4: Go to kaggle.com/datasets → slm-agents-results              ║
║  Step 5: Click "New Version" → upload the unzipped CSV/JSON files    ║
║                                                                      ║
║  Next session: attach slm-agents-results as a Dataset input          ║
║  session_init.py will auto-restore your checkpointed results         ║
╚══════════════════════════════════════════════════════════════════════╝
""".replace("{ts}", timestamp))


def save_model_cache(hf_cache_dir: str | Path | None = None) -> None:
    """
    ONE-TIME USE (Step 0 only): zip the HF model cache for upload as slm-agents-models.
    This is large (~20GB) — only run once after downloading all models.
    """
    if hf_cache_dir is None:
        hf_cache_dir = Path("/kaggle/working/hf_cache")
    hf_cache_dir = Path(hf_cache_dir)

    zip_path = Path("/kaggle/working/hf_cache_upload.zip")
    print(f"Zipping HF cache from {hf_cache_dir} ... (this may take a few minutes)")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(hf_cache_dir.rglob("*")):
            if f.is_file():
                arcname = f.relative_to(hf_cache_dir.parent)
                zf.write(f, arcname)

    size_gb = zip_path.stat().st_size / 1e9
    print(f"\n✅ HF cache zipped → {zip_path}  ({size_gb:.1f} GB)")
    print("""
Upload instructions:
  1. Download hf_cache_upload.zip from Kaggle output panel
  2. Unzip → you get a folder named hf_cache/
  3. Go to kaggle.com/datasets → create NEW dataset named "slm-agents-models"
  4. Upload the hf_cache/ folder contents
  5. Future sessions: attach slm-agents-models as input → models load instantly
""")
