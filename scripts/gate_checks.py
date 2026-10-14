"""Gate checks + shard merging, invoked by run_all.sh.

  python scripts/gate_checks.py g3   # merge step2 shards, verify baselines
  python scripts/gate_checks.py g4   # merge step3 shards, verify the sweep

A failed check exits non-zero and HALTS the run — the next stage's GPU hours
are never spent on broken premises.
"""

import _bootstrap  # noqa: F401  (must stay first)

import argparse
import sys

import pandas as pd

import common_run as cr

EXPECTED_N = {"hotpotqa": 300, "musique": 200}


def fail(msg: str) -> None:
    print(f"GATE FAIL: {msg}")
    sys.exit(1)


def g3() -> None:
    ep = cr.merge_shards("baseline_episodes", ["gpu0", "gpu1"], EXPECTED_N)
    cr.merge_shards("baseline_steps", ["gpu0", "gpu1"],
                    {"hotpotqa": 0, "musique": 0}, out_name="baseline_steps",
                    dup_extra_cols=["step_idx"])

    combos = set(zip(ep["policy"], ep["slm_name"], ep["dataset"]))
    expected = {
        ("always_slm_1b", "Qwen2.5-1.5B", "hotpotqa"),
        ("always_slm_1b", "Qwen2.5-1.5B", "musique"),
        ("always_slm_3b", "Qwen2.5-3B", "hotpotqa"),
        ("always_slm_3b", "Qwen2.5-3B", "musique"),
        ("always_llm", "Llama-3.1-8B", "hotpotqa"),
        ("always_llm", "Llama-3.1-8B", "musique"),
    }
    missing = expected - combos
    if missing:
        fail(f"missing baseline combos: {sorted(missing)}")

    # NaN check covers METRIC columns only. Text columns (predicted) are
    # legitimately empty when a model never answers — pandas reads "" as NaN,
    # and those rows are valid failures (EM=False), not corruption.
    metric_cols = ["success", "n_steps", "n_llm_calls", "n_slm_calls", "llm_call_ratio"]
    if ep[metric_cols].isna().any().any():
        bad = ep[metric_cols].columns[ep[metric_cols].isna().any()].tolist()
        fail(f"NaN values in metric columns: {bad}")
    n_empty = int(ep["predicted"].isna().sum())
    print(f"  empty predictions (model never answered): {n_empty}/{len(ep)}")

    for ds, n in EXPECTED_N.items():
        sub = ep[ep["dataset"] == ds]
        oracle = sub[sub["policy"] == "always_llm"]["success"].mean()
        slm1b = sub[sub["policy"] == "always_slm_1b"]["success"].mean()
        print(f"  {ds}: oracle SR={oracle:.3f} | always_slm_1b SR={slm1b:.3f}")
        if not oracle > slm1b:
            fail(f"H0 sanity violated on {ds}: oracle ({oracle:.3f}) <= "
                 f"always_slm_1b ({slm1b:.3f}). Investigate prompts/parser "
                 f"before spending sweep hours.")
        for pid, grp in sub.groupby("policy"):
            if len(grp) != n:
                fail(f"{pid}/{ds}: {len(grp)} rows != {n}")
    print("G3 OK")


def g4() -> None:
    df = cr.merge_shards("routing_episodes", ["gpu0", "gpu1"], EXPECTED_N)

    combos = df.groupby(["policy", "slm_name", "dataset"]).size()
    n_combos = len(combos)
    # conf 9 + entropy 8 + step 6 = 23 policies x 2 datasets (46) on 1.5B,
    # plus cascade 8 x 2 (16) on the 3-model label = 62
    if n_combos != 62:
        fail(f"{n_combos} routing combos != 62: {sorted(combos.index.tolist())}")
    if (combos < combos.index.get_level_values("dataset").map(EXPECTED_N)).any():
        short = combos[combos < combos.index.get_level_values("dataset").map(EXPECTED_N)]
        fail(f"underfilled combos: {short.to_dict()}")

    for ds in EXPECTED_N:
        mus = df[df["dataset"] == ds]
        if ds == "musique" and len(mus) == 0:
            fail("no musique rows at all — the resume-skip bug is back")

    # monotonicity (ties allowed): conf T↑ -> llm_ratio↑, entropy T↑ -> ↓
    def ratio(policy):
        return df[df["policy"] == policy]["llm_call_ratio"].mean()

    conf_ts = sorted(float(p.split("_T")[1]) for p in df["policy"].unique()
                     if p.startswith("conf_gate"))
    conf_rs = [ratio(f"conf_gate_T{t:.1f}") for t in conf_ts]
    ent_ts = sorted(float(p.split("_T")[1]) for p in df["policy"].unique()
                    if p.startswith("entropy_gate"))
    ent_rs = [ratio(f"entropy_gate_T{t:.1f}") for t in ent_ts]

    def monotone(vals, increasing=True):
        pairs = zip(vals, vals[1:])
        return all(b >= a if increasing else b <= a for a, b in pairs)

    if not monotone(conf_rs, increasing=True):
        print(f"  WARNING: conf_gate ratios not monotone: "
              f"{[f'{r:.3f}' for r in conf_rs]} (ties/noise tolerated)")
    if not monotone(ent_rs, increasing=False):
        print(f"  WARNING: entropy_gate ratios not monotone: "
              f"{[f'{r:.3f}' for r in ent_rs]} (ties/noise tolerated)")

    span_lo = min(conf_rs + ent_rs)
    span_hi = max(conf_rs + ent_rs)
    if not (span_lo < 0.5 < span_hi or span_lo == 0.0 or span_hi == 1.0):
        fail(f"LLM-call ratios do not span the dial: [{span_lo:.3f}, {span_hi:.3f}]")
    print(f"  conf_gate ratio span: {min(conf_rs):.3f} -> {max(conf_rs):.3f}")
    print(f"  entropy_gate ratio span: {min(ent_rs):.3f} -> {max(ent_rs):.3f}")
    print("G4 OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g3", "g4"])
    args = ap.parse_args()
    {"g3": g3, "g4": g4}[args.gate]()


if __name__ == "__main__":
    main()
