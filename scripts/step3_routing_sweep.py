"""Step 3 — routing sweep, sharded.

  gpu0: conf_gate (9 thresholds) + entropy_gate (8)   [1.5B + 8B resident]
  gpu1: step_gate (6 k-values) then cascade (8 (t1,t2) pairs)
        [step_gate: 1.5B + 8B; cascade: 1.5B + 3B + 8B ≈ 25GB on one card]

Fixes vs the notebook:
  * the done-skip inside a per-dataset loop is `continue`, never `return`
    (the notebook's `return` skipped MuSiQue for every resumed policy)
  * the cascade gets the same scratchpad answer fallback as run_react_episode
    (via common_run.run_inline_loop) so scoring is uniform

Output (per shard): results/routing_episodes.<shard>.csv
"""

import _bootstrap  # noqa: F401  (must stay first)

import argparse
import gc

import torch

import common_run as cr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", choices=["gpu0", "gpu1"], required=True)
    args = ap.parse_args()

    from src.models import load_model, get_model_by_name
    from src.routing import (
        ConfidenceGateRouter, EntropyGateRouter, StepCountRouter,
        ThreeLevelCascadeRouter,
        CONF_THRESHOLDS, ENTROPY_THRESHOLDS, STEP_KS, CASCADE_GRID,
    )
    from src.inference import generate_single_with_logits

    all_data = cr.load_all_datasets()
    expected_n = {ds: len(s) for ds, s in all_data.items()}

    out_path = cr.shard_path("routing_episodes", args.shard)
    kept_rows, done_keys = cr.load_done_keys(out_path, expected_n)
    print(f"[resume] {len(done_keys)} combos already complete on {args.shard}")

    slm_1b_cfg = get_model_by_name("Qwen2.5-1.5B")
    llm_cfg = get_model_by_name("Llama-3.1-8B")

    def run_gate_sweep(policy_id, router, slm_cfg, slm_model, slm_tok,
                       llm_model, llm_tok):
        nonlocal kept_rows
        for ds_name, samples in all_data.items():
            key = (policy_id, slm_cfg.name, ds_name)
            if key in done_keys:
                print(f"  skip {key} (done)")
                continue                      # fixed: was `return` in the notebook
            print(f"  [{policy_id}] {ds_name} ({len(samples)} q)")
            ep_rows, _ = cr.run_episode_loop(
                samples, ds_name, policy_id, slm_cfg, slm_model, slm_tok,
                llm_cfg, llm_model, llm_tok, router,
                kept_rows=kept_rows, out_path=out_path,
            )
            kept_rows = kept_rows + ep_rows
            done_keys.add(key)

    # ---- gpu0: conf_gate + entropy_gate (1.5B + 8B resident) -------------
    if args.shard == "gpu0":
        slm_model, slm_tok = load_model(slm_1b_cfg)
        llm_model, llm_tok = load_model(llm_cfg)
        print("1.5B + 8B resident")

        for T in CONF_THRESHOLDS:
            run_gate_sweep(f"conf_gate_T{T:.1f}", ConfidenceGateRouter(T),
                           slm_1b_cfg, slm_model, slm_tok, llm_model, llm_tok)
        for T in ENTROPY_THRESHOLDS:
            run_gate_sweep(f"entropy_gate_T{T:.1f}", EntropyGateRouter(T),
                           slm_1b_cfg, slm_model, slm_tok, llm_model, llm_tok)

        del slm_model, slm_tok, llm_model, llm_tok
        gc.collect()
        torch.cuda.empty_cache()

    # ---- gpu1: step_gate, then the 3-level cascade -----------------------
    else:
        slm_model, slm_tok = load_model(slm_1b_cfg)
        llm_model, llm_tok = load_model(llm_cfg)
        print("1.5B + 8B resident (step_gate)")

        for K in STEP_KS:
            run_gate_sweep(f"step_gate_K{K}", StepCountRouter(K),
                           slm_1b_cfg, slm_model, slm_tok, llm_model, llm_tok)

        # cascade adds the 3B (1.5B + 3B + 8B ≈ 25GB fp16 — fits easily)
        slm_3b_cfg = get_model_by_name("Qwen2.5-3B")
        slm_3b_m, slm_3b_t = load_model(slm_3b_cfg)
        print("3B added — cascade resident set complete")

        for t1, t2 in CASCADE_GRID:
            router = ThreeLevelCascadeRouter(t1, t2)
            policy_id = f"cascade_t1{t1:.1f}_t2{t2:.1f}"

            def make_gen(r):
                def gen(prompt):
                    out1 = generate_single_with_logits(slm_model, slm_tok, prompt)
                    if r.decide_level1(out1["first_token_entropy"]):
                        out2 = generate_single_with_logits(slm_3b_m, slm_3b_t, prompt)
                        if r.decide_level2(out2["first_token_entropy"]):
                            out3 = generate_single_with_logits(llm_model, llm_tok, prompt)
                            return out3["response"], True      # reached the 8B
                        return out2["response"], False
                    return out1["response"], False
                return gen

            for ds_name, samples in all_data.items():
                key = (policy_id, "cascade_1b_3b_8b", ds_name)
                if key in done_keys:
                    print(f"  skip {key} (done)")
                    continue
                print(f"  [{policy_id}] {ds_name} ({len(samples)} q)")
                ep_rows = cr.run_inline_loop(
                    samples, ds_name, policy_id, "cascade_1b_3b_8b",
                    make_gen(router),
                    kept_rows=kept_rows, out_path=out_path,
                )
                kept_rows = kept_rows + ep_rows
                done_keys.add(key)

        del slm_model, slm_tok, slm_3b_m, slm_3b_t, llm_model, llm_tok
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nSTEP3 {args.shard} OK — {len(done_keys)} combos complete")


if __name__ == "__main__":
    main()
