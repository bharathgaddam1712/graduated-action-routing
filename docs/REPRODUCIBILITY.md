# Reproducing the Study

## Recommended Workflow

Use `page2027_confirmatory/README.md` and the standalone launcher. Record the
Git commit before execution. Do not edit code, change dependency versions,
or change the GPU order/shard count during a run. Start a fresh output
directory if any of these change.

Training and calibration questions are separate from evaluation questions.
The new split builder rejects duplicate source IDs and normalized question
text across splits. Dataset exports are pinned. The first exploratory pilot
prefix is excluded, but the old pilot lacked a source revision, so historical
overlap cannot be certified solely from its IDs.

The corrected protocol fixes bracketed action parsing and records all routing
calls. The exploratory pipeline also used different generation and scoring
details. Evaluate every baseline under the confirmatory protocol; combining
old baseline scores with new routed scores would not be a valid comparison.

## CPU Reporting

Copy the bundle and the full completed run directory to a CPU machine:

```bash
cd page2027_confirmatory
python3.11 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
CUDA_VISIBLE_DEVICES="" python scripts/step7_review_stats.py --out runs/confirmatory
```

Keep SQLite episode stores, manifest/splits, router/calibration JSON, device
metadata, logs, and all report artifacts together. Publishing just aggregate
CSVs is insufficient to reproduce paired analyses.

## Historical Artifacts

`results/exploratory/` contains the merged historical baseline, routing, and
learned-router episodes, baseline steps, and fitted-router coefficients/AUC.
These support the current draft's exploratory claims. They are not new
confirmatory outputs. GPU-specific duplicate shards and smoke outputs are
not included in the public result snapshot.

`src/`, root `scripts/`, and `notebooks/` retain the older exploratory workflow.
The historical Kaggle runbook mentions an earlier workshop target and runtime
estimates. Use the confirmatory runbook for current execution.

`paper/overleaf_page2027/audit_figures.py` regenerates the exploratory paper
figures from `results/exploratory/`; it needs pandas, matplotlib, and
scikit-learn. The manuscript requires its own LaTeX compilation and page-limit
check. Confirmatory GPU execution and the manuscript PDF are not tested by
the CPU CI job.
