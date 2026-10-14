# Metric Definitions

Metrics are reported separately for each dataset and policy. All costs include
discarded small/intermediate candidates and confidence calls where applicable.
The reference for savings is the always-8B policy on the same question set.

| Metric | Definition / unit |
| --- | --- |
| `success` | Mean normalized single-gold exact match |
| `mean_alias_success` | Exact match against the gold answer or an available alias |
| `mean_llm_call_ratio` | Mean of per-episode executed 8B-step fractions |
| `mean_call_wall_s` | Mean sum of synchronized generation-call wall seconds per question |
| `mean_episode_wall_s` | Mean complete episode wall seconds, including retrieval and orchestration |
| `mean_cuda_elapsed_s` | Mean sum of CUDA-event elapsed intervals around generation |
| `mean_total_tokens` | Mean input plus output tokens over all model calls |
| `mean_parameter_b_tokens` | Sum of nominal model parameters in billions times input/output tokens, averaged per question |
| `peak_allocated_gib` | Maximum episode peak PyTorch allocated memory across the policy's questions |
| `measured_cost_per_question_seconds` | Alias for `mean_call_wall_s` |
| `compute_proxy_per_question` | Alias for `mean_parameter_b_tokens` |

For a metric `m`, savings are `1 - mean(m_policy) / mean(m_always_8B)`.
The raw summary stores fractions; formatted table columns show percentages.
A negative saving means a policy costs more than the reference. Savings are
computed separately for call latency, episode latency, total tokens, and the
parameter-billion-token compute proxy.

Parameter-times-token cost does not model prefill/decode differences, KV-cache
behavior, or attention complexity. It is not measured FLOPs. CUDA elapsed
time is not exclusive GPU active time. Token counts from different tokenizers
are workload descriptors rather than equivalent computational units.
The executed 8B-step fraction alone does not establish total compute savings.
Measured seconds are not dollars, billing, or energy.

Call wall time includes tokenization, generation, feature extraction, and
decoding. Episode wall time additionally includes the environment. Setup,
model loading, warm-up, and interrupted partial episodes are excluded from
completed-episode costs and documented separately where available.

Accuracy has Wilson 95% confidence intervals. Policy accuracy differences
use paired bootstrap resampling of question IDs with 2,000 repetitions.
Exact McNemar tests compare discordant paired outcomes; Holm adjustment
covers the displayed comparisons across both datasets. Random-policy seed
averages are computed within each question. Non-significance does not
establish equivalent accuracy. Latency/cost confidence intervals are not
currently generated.
