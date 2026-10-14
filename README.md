# Graduated Action Routing for ReAct Agents

Research code and paper artifacts for studying when a ReAct agent should
execute a small-model action or escalate to a larger model. The project targets
the PAgE 2027 workshop; the manuscript is a draft and the new confirmatory
GPU experiments have not yet been completed.

The recommended workflow is the independent **confirmatory pipeline** in
[`page2027_confirmatory/`](page2027_confirmatory/README.md). It evaluates
accuracy and measured inference cost on disjoint, frozen question splits.
Historical notebooks, source modules, and results are retained for inspecting
the exploratory study.

## Quick Start

Use a Linux server with Python 3.11 and CUDA-capable GPUs. Each GPU worker
loads the three FP16 models and requires at least 36 GiB free GPU memory.
Internet access and Hugging Face access to the gated Llama checkpoint are
needed for the first preparation stage.

```bash
git clone https://github.com/bharathgaddam1712/graduated-action-routing.git
cd graduated-action-routing/page2027_confirmatory
export HF_HOME=/scratch/$USER/hf-cache
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$HF_HOME" "$TMPDIR"
bash setup.sh
source .venv/bin/activate
huggingface-cli login

# Pilot, then resume the full evaluation with the same GPU list.
python -u run_server.py --gpus 1,0 --pilot 20
cat runs/confirmatory/runtime_estimate.json
python -u run_server.py --gpus 1,0
```

The pilot projects evaluation GPU time from observed episode times. Both GPU
workers run every policy on separate subsets of questions. Keep the GPU list
and shard count fixed when resuming. The launcher generates the final CPU
statistics after the evaluation is complete.

For an independent server upload, use the
[`page2027_confirmatory.zip`](downloads/page2027_confirmatory.zip) bundle.
Detailed installation, stages, cache configuration, and recovery commands
are in the [server runbook](page2027_confirmatory/README.md).

## Experiment Scope

| Component | Confirmatory configuration |
| --- | --- |
| Tasks | HotpotQA distractor and MuSiQue |
| Evaluation | 1,000 unique questions per dataset |
| Router training | 300 training questions per dataset |
| Calibration | 150 separate validation questions per dataset |
| Small model | Qwen2.5-1.5B-Instruct |
| Intermediate model | Qwen2.5-3B-Instruct |
| Large model | Llama-3.1-8B-Instruct |
| Precision | FP16 for all three models |
| Policies | 19 settings across direct, entropy, step, cascade, learned, confidence, query, and random policies |
| Main evaluation | 38,000 episodes |
| Statistical analysis | Wilson 95% CIs, question-paired bootstrap, exact McNemar tests, Holm correction |

Router training fits logistic regression on CPU from GPU-generated
trajectories. It does not update the language-model weights. Query and random
routing parameters are frozen on calibration data. Their achieved evaluation
cost mismatch is reported rather than assumed to be zero.

## Repository Layout

```text
page2027_confirmatory/       Recommended standalone experiment pipeline
  confirmatory/             Data, routing, inference, persistence, and reporting
  scripts/                  Step6 GPU experiments and step7 CPU statistics
  tests/                    Unit and mock end-to-end integration tests
  run_server.py             Single/dual-GPU orchestration
  setup.sh                  Pinned Linux GPU environment setup
  requirements*.txt         GPU runtime and CPU analysis dependencies
docs/                       Protocol, metrics, and contributor guidance
downloads/                  Independent server ZIP
paper/overleaf_page2027/    Audited LaTeX draft, bibliography, figures, audit notes
results/exploratory/         Historical merged results and fitted-router artifacts
src/                        Historical exploratory agent implementation
scripts/                    Historical step0-step5 server scripts
notebooks/                  Historical Kaggle notebooks and runbook
.github/workflows/          CPU verification in GitHub Actions
```

The latest manuscript entry point is
[`paper/overleaf_page2027/main.tex`](paper/overleaf_page2027/main.tex).
Its numerical results and figures currently describe the exploratory study.
Replace those claims with verified confirmatory results after the server run.

## Results and Metrics

Completed runs produce `page2027_confirmatory/runs/confirmatory/report/` with
per-question, per-call, and per-step traces; statistical comparisons;
error-bar plots; and CSV/LaTeX result tables. The report rejects incomplete
evaluation shards.

Metrics include exact match, alias-aware exact match, model usage, steps,
invalid actions, input/output tokens, per-call and episode latency, CUDA
generation elapsed time, peak allocated memory, and savings against always-8B.
See [metric definitions](docs/METRICS.md) for denominators, units, and limits.
Compute savings use an explicitly labeled parameter-times-token proxy.
Monetary cost, energy, and measured FLOPs are not currently collected.

## Verification and Reproducibility

The CPU suite checks routing behavior, discarded-call accounting, parser
behavior, split isolation, checkpoint identity, sharding, resume, and complete
mock report generation. A mock run is labeled in provenance and figures.
Real GPU compatibility must be checked with the included server smoke stage.

```bash
cd page2027_confirmatory
python -m pip install -r requirements-analysis.txt scikit-learn==1.6.1 rank-bm25==0.2.2
python -m unittest discover -s tests -v
```

Dataset revisions, resolved model revisions, question split hashes, code
hashes, dependency versions, router artifacts, calibration parameters, and
GPU metadata are recorded. Preserve the entire run directory for analysis.
See [reproducibility notes](docs/REPRODUCIBILITY.md).

## Paper and Reuse

The draft is not an accepted-publication record. Citation metadata will be
added when a stable preprint or publication identifier is available.
Third-party datasets and model checkpoints retain their own licenses and
access requirements. A project-wide code license has not yet been specified;
contact the repository owner before redistributing the code.
