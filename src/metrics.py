"""
Metrics for Paper B.

compute_task_success       — exact-match accuracy
compute_llm_call_ratio     — fraction of steps using LLM
compute_pareto_frontier    — task success vs LLM ratio for each policy × threshold
compute_step_distribution  — escalation rate by step index
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Episode-level metrics
# ---------------------------------------------------------------------------

def compute_task_success(episodes_df: pd.DataFrame) -> float:
    return float(episodes_df["success"].mean()) if len(episodes_df) > 0 else 0.0


def compute_llm_call_ratio(episodes_df: pd.DataFrame) -> float:
    return float(episodes_df["llm_call_ratio"].mean()) if len(episodes_df) > 0 else 0.0


def compute_summary(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (policy, slm_name, dataset) and compute:
      task_success, llm_call_ratio, mean_n_steps, n_episodes
    """
    rows = []
    for (policy, slm, ds), grp in episodes_df.groupby(["policy", "slm_name", "dataset"]):
        rows.append({
            "policy":          policy,
            "slm_name":        slm,
            "dataset":         ds,
            "task_success":    round(float(grp["success"].mean()), 4),
            "llm_call_ratio":  round(float(grp["llm_call_ratio"].mean()), 4),
            "mean_n_steps":    round(float(grp["n_steps"].mean()), 2),
            "n_episodes":      len(grp),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def compute_pareto_frontier(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (policy, slm_name, dataset) row in the summary, the pair
    (llm_call_ratio, task_success) is a point on the efficiency-accuracy plane.

    Returns the summary with a 'pareto_dominant' column:
    True if no other point has both lower LLM ratio AND higher task success.
    """
    df = summary_df.copy()
    df["pareto_dominant"] = False

    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            if (other["llm_call_ratio"] <= row["llm_call_ratio"] and
                    other["task_success"] >= row["task_success"] and
                    (other["llm_call_ratio"] < row["llm_call_ratio"] or
                     other["task_success"] > row["task_success"])):
                dominated = True
                break
        df.at[i, "pareto_dominant"] = not dominated

    return df


def extract_pareto_curve(summary_df: pd.DataFrame, policy: str, slm_name: str, dataset: str) -> pd.DataFrame:
    """Extract the Pareto curve points for one (policy, slm, dataset) across threshold sweep."""
    mask = (
        (summary_df["policy"].str.startswith(policy)) &
        (summary_df["slm_name"] == slm_name) &
        (summary_df["dataset"] == dataset)
    )
    return summary_df[mask].sort_values("llm_call_ratio").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step-level analysis
# ---------------------------------------------------------------------------

def compute_step_distribution(steps_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (policy, slm_name, dataset, step_idx), compute:
    - escalation_rate: fraction of steps at that index that used LLM
    - mean_entropy: mean first_token_entropy at that step
    - survival_rate: fraction of episodes still active at that step
    """
    rows = []
    for (policy, slm, ds, step_idx), grp in steps_df.groupby(
        ["policy", "slm_name", "dataset", "step_idx"]
    ):
        rows.append({
            "policy":         policy,
            "slm_name":       slm,
            "dataset":        ds,
            "step_idx":       step_idx,
            "escalation_rate": float((grp["model_used"] == "llm").mean()),
            "mean_entropy":   float(grp["first_token_entropy"].mean()),
            "n_steps":        len(grp),
        })
    df = pd.DataFrame(rows)
    # Baseline logs never escalate, so escalation_rate is 0 by construction;
    # survival (step counts are non-increasing, so max = episodes started)
    # is the position signal fig3 plots instead.
    df["survival_rate"] = df["n_steps"] / df.groupby(
        ["policy", "slm_name", "dataset"])["n_steps"].transform("max")
    return df


# ---------------------------------------------------------------------------
# Routing signal quality (for Fig 4)
# ---------------------------------------------------------------------------

def compute_routing_signal_auc(steps_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each routing signal (entropy, verbalized_conf, top1_prob),
    measure how well it predicts whether the SLM's step led to episode failure.

    AUC > 0.5 means the signal is informative for routing decisions.
    """
    label = (~steps_df["episode_success"]).astype(int)
    rows = []

    signals = {
        "first_token_entropy": steps_df.get("first_token_entropy"),
        "top1_prob_inverted":  1 - steps_df.get("top1_prob", 0),  # high prob = low uncertainty
        "verbalized_conf_inverted": steps_df.get("verbalized_conf").apply(
            lambda x: 1 - x if x is not None else 0.5
        ) if "verbalized_conf" in steps_df.columns else None,
    }

    for signal_name, signal_vals in signals.items():
        if signal_vals is None:
            continue
        valid = signal_vals.notna()
        if valid.sum() < 10:
            continue
        try:
            auc = roc_auc_score(label[valid], signal_vals[valid])
            rows.append({"signal": signal_name, "auc": round(auc, 3), "n": int(valid.sum())})
        except Exception:
            pass

    return pd.DataFrame(rows)
