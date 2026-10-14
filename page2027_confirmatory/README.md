# PAgE 2027 confirmatory experiments

Independent Linux/server bundle, prepared 2026-10-13. It does not import your
old repository or depend on the paper/Overleaf folder. Model weights and
datasets download on the server into your Hugging Face cache. They are not
included in the ZIP. A Hugging Face account with access to the Llama checkpoint
is required. No tokens are shipped in this bundle.

This trains a small logistic routing classifier on CPU using GPU-generated
trajectories. It does not fine-tune Qwen or Llama model weights. Most GPU time
is inference to collect trajectories and evaluate the frozen policies.

## Fixed experiment scope

| Family | Settings |
| --- | --- |
| Direct baselines | Always 1.5B, always 3B, always 8B |
| Entropy | 1.0, 1.5, 2.0, 2.5 nats |
| Step gate | K1, K2 (zero-based action indices) |
| Cascade | (0.5, 2.0), (1.0, 2.5) nats |
| Learned logit router | 0.3, 0.4, 0.5 |
| Prompted confidence | 0.9 |
| Query-level gate | One calibration-frozen initial entropy threshold per dataset |
| Random step router | One calibration-frozen probability, seeds 17, 29, 43 |

19 settings x (1,000 HotpotQA + 1,000 MuSiQue) = **38,000 evaluation episodes**.
The two cascade settings were selected from the exploratory low-8B region;
the held-out study tests them without choosing new settings on evaluation data.

Per dataset: 300 unique questions from the official training split for router
training, 150 validation questions for calibration, and 1,000 different
validation questions for evaluation. Text and source IDs cannot overlap across
the new splits. The first 300 HotpotQA / 200 MuSiQue validation questions from
the pinned exports are excluded before sampling to avoid the original pilot
prefix. This exclusion assumes the pilot used the same source ordering; the
old run did not preserve a source revision, so historical overlap cannot be
certified solely from its synthetic IDs. New split isolation is enforced.

The workflow performs 600 training episodes plus 300 held-out router
calibration episodes. Cost calibration adds at most 3,600 episodes: one
entropy reference, up to five query thresholds, and six random probabilities
on the 300 calibration questions. Smoke adds 38 episodes. Total is at most
**42,538 episodes**; a pilot is a prefix of evaluation and is reused on resume.
Dataset/source revisions, model revisions, split content hash, code hash,
dependency versions, and frozen calibration/router artifacts are recorded.

## Server commands

Put `page2027_confirmatory.zip` in `/scratch/user2/`, then run:

```bash
mkdir -p /scratch/user2/page2027 /scratch/user2/tmp /scratch/user2/hf-cache
cd /scratch/user2/page2027
unzip /scratch/user2/page2027_confirmatory.zip
cd page2027_confirmatory
export HF_HOME=/scratch/user2/hf-cache
export TMPDIR=/scratch/user2/tmp
export PIP_CACHE_DIR=/scratch/user2/pip-cache
bash setup.sh
source .venv/bin/activate
huggingface-cli login
```

Accept the model license first:
https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
If Python 3.11 is unavailable but Python 3.12 is installed:
`PYTHON=python3.12 bash setup.sh`. Do not use Python 3.14 with these pinned
server packages. Python 3.10 is also accepted.

The setup installs torch 2.7.1 from the official CUDA 12.8 wheel index,
separately from PyPI packages. Your driver does not need a matching local CUDA
toolkit installation. Every GPU worker runs an actual warm-up generation on
all three models, which exposes CUDA/attention/tokenizer incompatibility
before the main evaluation. Approximate FP16 model weights are 25 GB per GPU;
the backend requires at least 36 GiB free and exposes exactly one GPU per
process. No model parallelism, FlashAttention build, or bitsandbytes is needed.

Your supplied snapshot shows another compute job on GPU0. Start with GPU1:

```bash
python -u run_server.py --gpus 1 --pilot 20
cat runs/confirmatory/runtime_estimate.json
python -u run_server.py --gpus 1
```

The first command prepares the data/models, trains the router, freezes cost
calibration, performs the smoke checks, and evaluates the first 20 questions
per dataset for every policy. It then projects remaining evaluation GPU time
from measured episode times. The full command skips completed work and
continues with the remaining questions.

If both GPUs are free of other compute jobs, use this pair of commands instead:

```bash
python -u run_server.py --gpus 1,0 --pilot 20
python -u run_server.py --gpus 1,0
```

Each GPU evaluates all policies on a disjoint subset of questions. A question's
baseline and routed policies run on the same GPU. Keep the same GPU list and
shard count from pilot through completion. Use a new `--out runs/another_run`
if changing GPU count, code, dependencies, or dataset settings. Concurrent
unrelated GPU workloads invalidate clean timing comparisons even if memory
is available; the program does not stop or modify anyone else's processes.

For a terminal-independent run:

```bash
nohup python -u run_server.py --gpus 1,0 > runs/confirmatory/launcher.log 2>&1 &
tail -f runs/confirmatory/logs/shard-0.log
```

Run the pilot first so its directories exist. `tmux` is another option.
On an interruption, rerun the identical command. SQLite commits one complete
episode at a time. A partially executed episode is rerun; completed episodes
are reused without duplicates. Linux advisory locks prevent simultaneous
writers to the same shard. If a worker fails, the launcher stops its other
workers and returns an error. No scientific report is produced for an
incomplete evaluation.

## Results to bring back

Final output is `runs/confirmatory/report/`:

- `summary.csv`: exact match, Wilson intervals, executed 8B-step fraction,
  actual per-tier invocation counts, tokens, latency savings, token savings,
  compute-proxy savings, per-question measured cost, and two frontiers.
- `paired_tests.csv`: question-paired differences and bootstrap intervals,
  exact McNemar tests, and Holm correction across all displayed comparisons.
- `random_seed_average.csv`: random results averaged within each question;
  the three seeds are not treated as 3,000 independent questions.
- `episodes.csv`, `calls.csv`, `steps.csv`: all completed evaluation traces
  and measured costs, including discarded candidate/confidence/mid-tier calls.
- `provenance.json`: frozen protocol, splits/model identities, router AUCs,
  cost calibration, and interpretation notes.
- `confirmatory_results_table.csv` and `.tex`: an editable final results table.
- `hotpotqa_*_with_ci` and `musique_*_with_ci`: PNG/PDF figures with Wilson
  error bars for measured cost and executed 8B-step fraction.
- `analysis_versions.json`: versions of the CPU statistics/plotting packages.

Archive the entire `runs/confirmatory` directory, not just the summary: the
SQLite files contain training/calibration traces, and the device JSON files
contain hardware and load/warm-up timing. Report the actual query/random cost
mismatch against entropy T2.0 on evaluation data. Calibration is approximate;
the code makes no exact held-out cost-matching claim. It does not retune using
evaluation outcomes.

Primary success matches the pilot's single-gold normalization. A secondary
alias-aware score is recorded. Supporting-fact F1 is not evaluated. Confidence
intervals are descriptive, not equivalence certificates; non-significance
does not establish equal performance. Ordinary bootstrap CIs are not
multiplicity-adjusted; the displayed exact paired-test p-values are.

## Cost and environment definitions

Per-call synchronized wall time includes input tokenization, generation,
feature extraction, and decoding. CUDA events measure the elapsed interval
around `generate()`. These are actual observed timing measurements, not
FLOP counts, exclusive GPU active time, energy, or billing. Per-model input and
output tokens are logged. Parameter-billion x tokens is a clearly named
compute proxy, not measured FLOPs, energy, or monetary cost. The report derives
latency, token, and compute-proxy savings relative to always-8B. Resident model
memory and per-episode peak allocation are recorded. Model loading/warm-up times are separate; cost
totals cover completed episodes and exclude interrupted partial work.

This is a corrected confirmatory environment: bracketed Search inputs are
parsed correctly, malformed actions get an explicit observation, and an
unfinished episode has an empty predicted answer (no scratchpad fallback).
Direct baselines generate only once per step. Logit features use FP32 softmax
and entropy, rather than the pilot's FP16 probability computation. Chat-template
special tokens are added exactly once. Keep this protocol consistent across
all new policies and report the confirmatory results separately from the old
exploratory table. Policy selection must not be changed after seeing the new
evaluation scores.

## Individual stage interface

```bash
python scripts/step6_confirmatory_review.py --help
python scripts/step6_confirmatory_review.py --stage prepare --out runs/confirmatory
CUDA_VISIBLE_DEVICES=1 python scripts/step6_confirmatory_review.py --stage fit --out runs/confirmatory
CUDA_VISIBLE_DEVICES=1 python scripts/step6_confirmatory_review.py --stage calibrate --out runs/confirmatory
CUDA_VISIBLE_DEVICES=1 python scripts/step6_confirmatory_review.py --stage smoke --out runs/confirmatory
CUDA_VISIBLE_DEVICES=1 python scripts/step6_confirmatory_review.py --stage run --shard 0 --num-shards 1 --out runs/confirmatory
python scripts/step7_review_stats.py --out runs/confirmatory
```

Use `run_server.py` for ordinary operation. Its defaults freeze the required
1,000+1,000 evaluation size. A mock run requires `--mock` at preparation and
its provenance says MOCK. Mock output is only for tests and must never enter
the paper. Local CPU tests can run without torch or model weights:

```bash
python -m unittest discover -s tests -v
```

## CPU-only statistics

All step7 work is CPU-only: Wilson intervals, question-paired bootstrap,
exact McNemar tests, Holm adjustment, figures, and the final table. The full
launcher runs it automatically after successful evaluation. To regenerate
it on a separate CPU machine, copy this bundle and the entire run directory:

```bash
python3.11 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
CUDA_VISIBLE_DEVICES="" python scripts/step7_review_stats.py --out runs/confirmatory
```

No torch, model weights, CUDA installation, or GPU is needed for step7.
Reporting verifies the frozen code and data hashes, but does not require the
CPU machine to match the GPU runtime's dependency versions. It refuses an
incomplete evaluation instead of publishing a partial confirmatory table.

## Verification limits and references

The bundle is tested locally for CPU logic and a mock end-to-end workflow.
Real GPU generation and the pinned server package stack require the server's
smoke test; no universal "will not break" guarantee is possible. Internet
access, gated-model approval, disk space, and the server Python/CUDA runtime
remain prerequisites. The pilot produces a measured runtime estimate; the
earlier 35-45 GPU-hour planning figure is not a verified estimate for this
corrected 42,538-episode pipeline.

Official installation instructions:
https://pytorch.org/get-started/previous-versions/#v271
Generation interface:
https://huggingface.co/docs/transformers/main_classes/text_generation
Source data:
https://huggingface.co/datasets/hotpotqa/hotpot_qa
https://huggingface.co/datasets/dgslibisey/musique
