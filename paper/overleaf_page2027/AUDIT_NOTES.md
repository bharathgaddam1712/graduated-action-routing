# Draft audit, 2026-10-12

Scope: the complete Overleaf main.tex, all 14 bibliography entries, all five
figures, the three merged episode CSVs, baseline step logs, and the source
paths responsible for generation, routing, retrieval, scoring, and plotting.
This is a claim/evidence audit, not a guarantee of acceptance or a plagiarism
certificate. Bibliographic identities and the high-level related-work
summaries were checked against primary paper records and abstracts. It is
not a full-text replication of all 14 cited papers.

## Corrections

- Corrected the Hybrid LLM author list against arXiv:2404.14618.
- Replaced generic model-report author attributions with Qwen et al. and
  Grattafiori et al., following the current arXiv records.
- Added primary-source links to all bibliography entries and protected titles.
- Added previously missing Qwen, Llama, and BM25 citations at their uses.
- Distinguished calibrated verbalized uncertainty in Lin et al. from our
  prompted confidence gate; corrected request-level cascade wording.
- Changed seven policy families to six (41 configurations including baselines).
- Removed formal preregistration wording: project notes contain hypotheses,
  but no independently verified timestamped registration was established.
- Removed unpaired two-proportion p-values and equivalence language.
- Replaced oracle/ceiling language with an always-8B reference baseline.
- Defined the mean of episode-level step fractions. It is not the pooled
  step fraction, a relative call count, or measured total compute savings.
- Disclosed candidate regeneration, confidence calls, intermediate 3B calls,
  and duplicate generation in the always-8B runner.
- Disclosed the first-N question selection, input/output limits, 500-character
  retrieved observation truncation, and single-answer exact-match scoring.
- Disclosed training-set AUC, reused training/evaluation questions, and
  exploratory threshold selection. MuSiQue is not held-out task transfer.
- Softened causal interpretations of first-token entropy. Low entropy can
  reflect response formatting and does not certify correct actions.
- Clarified that Figure 4 pools 6,418 steps from 1.5B, 3B, and 8B models;
  the learned router separately trains on 1,859 1.5B steps.
- Removed embedded figure numbers and unsupported plot titles. LaTeX now
  assigns numbers in document order. Included the 3B baseline in sweep plots.
- Completed a follow-up wording pass: specified the exact threshold grids,
  clarified missing-confidence escalation, replaced remaining relative-call
  wording, cited the introductory agent description, and explicitly referenced
  all five figures. Cascade captions now give exact observed scores rather
  than near-parity language. No separate 3B execution counts are claimed.
- Recomputed dominance separately by dataset; the production metrics helper
  compares datasets globally and should be corrected before future analysis.
  Experiment code and raw result files were not edited during this audit.

## Evidence map

| Claim | Local evidence |
| --- | --- |
| Baseline scores | results/baseline_episodes.csv |
| Threshold and cascade scores | results/routing_episodes.csv |
| Online learned-router scores | results/learned_router_episodes.csv |
| Entropy-by-step and pooled signal AUC | results/baseline_steps.csv |
| Training AUC 0.7060, step ablation 0.7194 | results/learned_router_auc.txt; scripts/step4_learned_router.py |
| Candidate-first and confidence calls | src/react_env.py |
| Cascade and learned generation | scripts/step3_routing_sweep.py; scripts/step4_learned_router.py |
| Duplicate always-8B generation | scripts/step2_baselines.py; src/react_env.py |
| Mean per-episode ratio | src/metrics.py; scripts/common_run.py |
| First-N selection and BM25 | src/datasets.py |
| Token limits, greedy decoding | src/inference.py |

## Remaining limitations

Audit script passed: 20,500 episodes; 82 dataset-policy configurations;
6,418 baseline steps; 1,859 router-training steps; training AUCs
0.7060106209 and 0.7193757973; pooled signal AUCs 0.6583803355 and
0.6557439888; all 14 bibliography keys cited and resolved; all learned-router
settings dominated within their own datasets; no tested policy meets H1.

- Held-out questions and episode-grouped router validation are still needed.
- Paired confidence intervals/tests and actual GPU/token cost measurements
  are still needed before stronger accuracy or efficiency claims.
- The offline signal plot pools models with different tokenizers and
  accuracies; model-specific held-out analysis would be more informative.
- Source consistency does not prove exact historical execution provenance:
  environment versions and the code revision used for the GPU runs should
  be preserved with the final artifact.
- The native compiler failed with "Unable to find standard directories for
  platform". PDF layout, bibliography rendering, and the eight-page body
  limit remain unverified; compile in Overleaf before submission.

## Review numbering and workshop status

The `review` class option adds margin line numbers. PAgE's published CFP
requires anonymity and SIGPLAN two-column format, with up to eight pages
excluding bibliography; it does not explicitly require red line numbers.
Keeping them is useful for review. Remove `review` for a clean reading copy,
and retain `anonymous` for submission.

The CFP also states accepted submissions are presented at the workshop and
do not appear in the ACM Digital Library.

Sources: https://popl27.sigplan.org/home/page-2027 and
https://www.sigplan-pages.sigplan.hosting.acm.org/Resources/Author/
