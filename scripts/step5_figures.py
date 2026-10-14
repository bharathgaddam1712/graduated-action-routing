"""Step 5 — analysis, 5 paper figures, hypothesis verification. CPU-only.

Guards: missing learned-router results (fallback path) or empty signal table
degrade gracefully — figures that can render, do; the run never crashes on an
empty Pareto region.
"""

import _bootstrap  # noqa: F401  (must stay first)

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import common_run as cr

POLICY_STYLES = {
    "conf_gate":    ("#1f77b4", "o", "Conf gate"),
    "entropy_gate": ("#d62728", "s", "Entropy gate"),
    "step_gate":    ("#ff7f0e", "^", "Step gate"),
    "cascade":      ("#2ca02c", "D", "3-Level cascade"),
    "learned":      ("#9467bd", "P", "Learned router"),
}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150,
                     "lines.linewidth": 2, "lines.markersize": 7})


def main() -> None:
    from src.metrics import (
        compute_summary, compute_pareto_frontier, compute_step_distribution,
        compute_routing_signal_auc,
    )

    results = cr.results_dir()
    figdir = cr.repo_root() / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name in ["baseline_episodes", "routing_episodes", "learned_router_episodes"]:
        p = results / f"{name}.csv"
        if not p.exists():
            print(f"[step5] WARNING: {p} missing — skipping its curves")
            continue
        df = pd.read_csv(p)
        if len(df):
            frames.append(df)
    if not frames:
        raise SystemExit("[step5] no result CSVs found — run steps 2-4 first")
    all_ep = pd.concat(frames, ignore_index=True)

    steps_path = results / "baseline_steps.csv"
    steps_df = pd.read_csv(steps_path) if steps_path.exists() else pd.DataFrame()

    summary = compute_pareto_frontier(compute_summary(all_ep))
    step_dist = (compute_step_distribution(steps_df)
                 if len(steps_df) else pd.DataFrame())
    print(f"Total episodes: {len(all_ep)} | policies: {all_ep['policy'].nunique()}")

    def curves(ax, ds, slm=None):
        for prefix, (color, marker, label) in POLICY_STYLES.items():
            mask = (summary["policy"].str.startswith(prefix)) & (summary["dataset"] == ds)
            if slm is not None:
                mask &= summary["slm_name"].isin([slm, "cascade_1b_3b_8b"])
            pts = summary[mask].sort_values("llm_call_ratio")
            if len(pts):
                ax.plot(pts["llm_call_ratio"], pts["task_success"],
                        color=color, marker=marker, label=label, alpha=0.8)

    def anchors(ax, ds):
        base = summary[summary["dataset"] == ds]
        for pid, label, m in [("always_slm_1b", "Always SLM", "x"),
                              ("always_llm", "Always LLM (Oracle)", "*")]:
            r = base[base["policy"] == pid]
            if len(r):
                ax.scatter(r["llm_call_ratio"], r["task_success"], marker=m,
                           s=150, color="black", zorder=5, label=label)

    # FIG 1 — Pareto, HotpotQA
    fig, ax = plt.subplots(figsize=(8, 6))
    curves(ax, "hotpotqa", slm="Qwen2.5-1.5B")
    anchors(ax, "hotpotqa")
    ax.set_xlabel("LLM Call Ratio (fraction of steps using LLM)")
    ax.set_ylabel("Task Success Rate (Exact Match)")
    ax.set_title("Figure 1: Pareto Frontier — HotpotQA (SLM: Qwen2.5-1.5B)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_pareto_frontier.png", bbox_inches="tight")
    plt.close(fig)
    print("fig1 saved")

    # FIG 2 — 3-level vs 2-level, HotpotQA
    fig, ax = plt.subplots(figsize=(8, 5))
    for prefix, color, label in [("entropy_gate", "#d62728", "Entropy gate (2-level)"),
                                 ("cascade", "#2ca02c", "3-Level cascade")]:
        pts = summary[(summary["policy"].str.startswith(prefix)) &
                      (summary["dataset"] == "hotpotqa")].sort_values("llm_call_ratio")
        if len(pts):
            ax.plot(pts["llm_call_ratio"], pts["task_success"],
                    color=color, marker="o", label=label)
    r_oracle = summary[(summary["policy"] == "always_llm") &
                       (summary["dataset"] == "hotpotqa")]
    if len(r_oracle):
        ax.axhline(float(r_oracle["task_success"].values[0]),
                   color="gray", linestyle="--", label="Oracle ceiling")
    ax.set_xlabel("LLM Call Ratio")
    ax.set_ylabel("Task Success Rate")
    ax.set_title("Figure 2: 3-Level Cascade vs Best 2-Level Policy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_3level_vs_2level.png", bbox_inches="tight")
    plt.close(fig)
    print("fig2 saved")

    # FIG 3 — step-level signals (needs baseline_steps.csv)
    if len(step_dist):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, ds in zip(axes, ["hotpotqa", "musique"]):
            sub = step_dist[(step_dist["dataset"] == ds) &
                            (step_dist["slm_name"] == "Qwen2.5-1.5B")]
            if not len(sub):
                ax.set_title(f"Figure 3: Step-Level Signals — {ds} (no data)")
                continue
            ax.bar(sub["step_idx"], sub["mean_entropy"], color="#1f77b4",
                   alpha=0.7, label="Mean first-token entropy")
            ax2 = ax.twinx()
            ax2.plot(sub["step_idx"], sub["survival_rate"],
                     color="#d62728", marker="o", label="Episodes still active")
            ax2.set_ylim(0, 1.05)
            ax.set_xlabel("ReAct Step Index")
            ax.set_ylabel("Mean First-Token Entropy")
            ax2.set_ylabel("Fraction of Episodes Still Active")
            ax.set_title(f"Figure 3: Step-Level Signals — {ds.title()}")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right",
                      framealpha=0.9)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "fig3_step_escalation.png", bbox_inches="tight")
        plt.close(fig)
        print("fig3 saved")
    else:
        print("fig3 SKIPPED (no baseline_steps.csv)")

    # FIG 4 — routing signal AUC (guards empty table)
    signal_aucs = (compute_routing_signal_auc(steps_df)
                   if len(steps_df) else pd.DataFrame())
    print("Routing signal AUC:")
    print(signal_aucs.to_string(index=False) if len(signal_aucs) else "  (none)")
    if len(signal_aucs):
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#1f77b4", "#d62728", "#2ca02c"]
        ax.bar(signal_aucs["signal"], signal_aucs["auc"],
               color=colors[:len(signal_aucs)], alpha=0.8)
        ax.axhline(0.5, color="gray", linestyle="--", label="Random baseline")
        ax.set_ylabel("AUC (predicting SLM failure)")
        ax.set_title("Figure 4: Routing Signal Quality")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "fig4_signal_quality.png", bbox_inches="tight")
        plt.close(fig)
        print("fig4 saved")
    else:
        print("fig4 SKIPPED (no signal AUC rows)")

    # FIG 5 — cross-task, MuSiQue
    fig, ax = plt.subplots(figsize=(8, 6))
    curves(ax, "musique")
    ax.set_xlabel("LLM Call Ratio")
    ax.set_ylabel("Task Success Rate")
    ax.set_title("Figure 5: Cross-Task Generalization — MuSiQue")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "fig5_cross_task.png", bbox_inches="tight")
    plt.close(fig)
    print("fig5 saved")

    # ---- hypothesis verification ------------------------------------------
    print("\n=== HYPOTHESIS VERIFICATION ===")
    base_hp = summary[summary["dataset"] == "hotpotqa"]

    def sr_of(pid):
        r = base_hp[base_hp["policy"] == pid]
        return float(r["task_success"].values[0]) if len(r) else None

    oracle_sr, slm_sr = sr_of("always_llm"), sr_of("always_slm_1b")
    if oracle_sr is not None and slm_sr is not None:
        print(f"H0 — oracle SR={oracle_sr:.3f} > SLM SR={slm_sr:.3f}: "
              f"{oracle_sr > slm_sr}")
    else:
        print("H0 — cannot evaluate (missing baseline rows)")

    if oracle_sr is not None:
        routing_pts = base_hp[~base_hp["policy"].isin(
            ["always_slm_1b", "always_slm_3b", "always_llm"])]
        good = routing_pts[(routing_pts["task_success"] >= 0.9 * oracle_sr) &
                           (routing_pts["llm_call_ratio"] < 0.6)]
        print(f"H1 — >=90% oracle SR at <60% LLM calls: {len(good) > 0}")
        if len(good):
            best = good.loc[good["llm_call_ratio"].idxmin()]
            print(f"     best: {best['policy']} SR={best['task_success']:.3f} "
                  f"LLM_ratio={best['llm_call_ratio']:.3f}")
        pareto = routing_pts[routing_pts["pareto_dominant"]]["policy"].unique()
        print("H2 — uncertainty-based routing on Pareto frontier: "
              f"entropy={any('entropy_gate' in p for p in pareto)}, "
              f"learned={any('learned' in p for p in pareto)}, "
              f"step_gate={any('step_gate' in p for p in pareto)}")

    print("\nSTEP5 OK — figures in", figdir)


if __name__ == "__main__":
    main()
