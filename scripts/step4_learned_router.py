"""Step 4 — learned router: train logreg on step-level logit features from the
always_slm_1b baseline, then sweep escalation thresholds on both datasets.

Single process (one GPU). Training itself is CPU and takes seconds.
Outputs: results/learned_router_episodes.csv, learned_router_auc.txt,
         learned_router_coefs.json
"""

import _bootstrap  # noqa: F401  (must stay first)

import argparse
import gc
import json

import torch

import common_run as cr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()

    from src.models import load_model, get_model_by_name
    from src.routing import (
        train_learned_router, LEARNED_THRESHOLDS,
    )
    from src.inference import generate_single_with_logits

    results = cr.results_dir()
    steps_csv = results / "baseline_steps.csv"
    if not steps_csv.exists():
        raise SystemExit("[step4] results/baseline_steps.csv missing — "
                         "run step2 + merge first")

    import pandas as pd
    steps_df = pd.read_csv(steps_csv)
    steps_1b = steps_df[steps_df["slm_name"] == "Qwen2.5-1.5B"]
    print(f"Training data: {len(steps_1b)} steps from always_slm_1b baseline")

    learned_router, auc, coefs = train_learned_router(steps_1b, verbose=True)

    (results / "learned_router_auc.txt").write_text(f"AUC={auc:.4f}\n")
    with open(results / "learned_router_coefs.json", "w") as f:
        json.dump({"auc": auc, "coefs": {k: float(v) for k, v in coefs.items()}},
                  f, indent=2)

    # Trajectory-conditioning ablation (notebook cell 4): does step_idx help?
    from src.routing import LOGIT_FEATURES_WITH_STEP
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    import numpy as np

    feats_step = [f for f in LOGIT_FEATURES_WITH_STEP if f in steps_1b.columns]
    df_abl = steps_1b.dropna(subset=feats_step)
    X_abl = df_abl[feats_step].values
    y_abl = (~df_abl["episode_success"]).astype(int).values
    clf_step = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
    ])
    clf_step.fit(X_abl, y_abl)
    auc_step = (roc_auc_score(y_abl, clf_step.predict_proba(X_abl)[:, 1])
                if len(np.unique(y_abl)) > 1 else 0.5)
    delta = auc_step - auc
    print(f"AUC logit-only={auc:.3f} | +step_idx={auc_step:.3f} | delta={delta:+.3f}")
    with open(results / "learned_router_auc.txt", "w") as f:
        f.write(f"AUC_logit_only={auc:.4f}\nAUC_with_step={auc_step:.4f}\n"
                f"delta={delta:+.4f}\n")

    # Pre-registered fallback: AUC <= 0.60 → skip the sweep, record the
    # negative result, exit 0 (the paper reports the cascade instead).
    if auc <= 0.60:
        print(f"AUC={auc:.3f} <= 0.60 — LEARNED ROUTER FALLBACK: writing empty "
              f"results file and stopping. Report as negative result.")
        lr_path = results / "learned_router_episodes.csv"
        if not lr_path.exists():
            pd.DataFrame(columns=[
                "question_id", "dataset", "policy", "slm_name", "success",
                "predicted", "gold_answer", "n_steps", "n_llm_calls",
                "n_slm_calls", "llm_call_ratio",
            ]).to_csv(lr_path, index=False)
        with open(results / "learned_router_auc.txt", "a") as f:
            f.write("fallback_triggered=true\n")
        print("STEP4 OK (fallback path)")
        return

    all_data = cr.load_all_datasets()
    expected_n = {ds: len(s) for ds, s in all_data.items()}

    lr_path = results / "learned_router_episodes.csv"
    kept_rows, done_keys = cr.load_done_keys(lr_path, expected_n)
    print(f"[resume] {len(done_keys)} combos already complete")

    slm_cfg = get_model_by_name("Qwen2.5-1.5B")
    llm_cfg = get_model_by_name("Llama-3.1-8B")
    slm_m, slm_t = load_model(slm_cfg)
    llm_m, llm_t = load_model(llm_cfg)
    print("1.5B + 8B resident")

    for T in LEARNED_THRESHOLDS:
        learned_router.set_threshold(T)
        policy_id = f"learned_T{T:.1f}"

        def gen(prompt, r=learned_router):
            slm_out = generate_single_with_logits(slm_m, slm_t, prompt)
            if r.decide_from_features(slm_out):
                llm_out = generate_single_with_logits(llm_m, llm_t, prompt)
                return llm_out["response"], True
            return slm_out["response"], False

        for ds_name, samples in all_data.items():
            key = (policy_id, slm_cfg.name, ds_name)
            if key in done_keys:
                print(f"  skip {key} (done)")
                continue
            print(f"  [{policy_id}] {ds_name} ({len(samples)} q)")
            ep_rows = cr.run_inline_loop(
                samples, ds_name, policy_id, slm_cfg.name, gen,
                kept_rows=kept_rows, out_path=lr_path,
                max_steps=args.max_steps,
            )
            kept_rows = kept_rows + ep_rows
            done_keys.add(key)

    del slm_m, slm_t, llm_m, llm_t
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nSTEP4 OK — {len(done_keys)} combos complete, AUC={auc:.3f}")


if __name__ == "__main__":
    main()
