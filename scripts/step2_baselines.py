"""Step 2 — baseline runs, sharded.

  gpu0: always_slm_1b, always_slm_3b (both datasets; AlwaysSLM never
        escalates, so the SLM is passed as the llm dummy — the 8B is never
        loaded on this shard)
  gpu1: always_llm (both datasets; SLM arg = the 8B itself)

Outputs (per shard): results/baseline_episodes.<shard>.csv,
                     results/baseline_steps.<shard>.csv
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
    from src.routing import AlwaysSLM, AlwaysLLM

    all_data = cr.load_all_datasets()
    expected_n = {ds: len(s) for ds, s in all_data.items()}

    slm_1b = get_model_by_name("Qwen2.5-1.5B")
    slm_3b = get_model_by_name("Qwen2.5-3B")
    llm = get_model_by_name("Llama-3.1-8B")

    if args.shard == "gpu0":
        combos = [("always_slm_1b", AlwaysSLM(), slm_1b),
                  ("always_slm_3b", AlwaysSLM(), slm_3b)]
    else:
        combos = [("always_llm", AlwaysLLM(), llm)]

    ep_path = cr.shard_path("baseline_episodes", args.shard)
    step_path = cr.shard_path("baseline_steps", args.shard)

    kept_eps, done_keys = cr.load_done_keys(ep_path, expected_n)
    print(f"[resume] {len(done_keys)} combos already complete on {args.shard}")

    # Step rows survive only for complete combos (same dedupe rule).
    kept_steps = []
    if step_path.exists() and done_keys:
        import pandas as pd
        sdf = pd.read_csv(step_path)
        mask = sdf.apply(lambda r: (r["policy"], r["slm_name"], r["dataset"]) in done_keys, axis=1)
        kept_steps = sdf[mask].to_dict("records")

    for policy_id, router, slm_cfg in combos:
        print(f"\n=== {policy_id} | {slm_cfg.name} ({args.shard}) ===")
        slm_model, slm_tok = load_model(slm_cfg)
        # always_llm: the 8B plays both roles (as in the notebook). always_slm:
        # router never escalates, so the SLM doubles as the llm dummy and the
        # 8B is never loaded on this shard.
        llm_model, llm_tok, llm_cfg = slm_model, slm_tok, slm_cfg

        for ds_name, samples in all_data.items():
            key = (policy_id, slm_cfg.name, ds_name)
            if key in done_keys:
                print(f"  skip {key} (done)")
                continue
            print(f"  {ds_name} ({len(samples)} q)")
            ep_rows, step_rows = cr.run_episode_loop(
                samples, ds_name, policy_id, slm_cfg, slm_model, slm_tok,
                llm_cfg, llm_model, llm_tok, router,
                kept_rows=kept_eps, out_path=ep_path,
                kept_step_rows=kept_steps, steps_path=step_path,
            )
            kept_eps = kept_eps + ep_rows
            kept_steps = kept_steps + step_rows
            done_keys.add(key)

        del slm_model, slm_tok
        gc.collect()
        torch.cuda.empty_cache()
        print("  GPU freed")

    print(f"\nSTEP2 {args.shard} OK — {len(done_keys)} combos complete")


if __name__ == "__main__":
    main()
