"""
common_run.py — shared machinery for the step scripts.

  * resume with partial-combo dedupe (a combo is (policy, slm_name, dataset);
    incomplete combos are DROPPED from the loaded CSV so a re-run can never
    duplicate rows — the Kaggle notebooks kept them and double-counted)
  * per-shard output paths (two processes never write the same CSV)
  * shard merge with uniqueness/count asserts
  * the episode loops (baseline/router/cascade/learned) with periodic atomic
    checkpointing and the answer-extraction fallback applied UNIFORMLY
    (the inline cascade/learned cells in the notebooks lacked the scratchpad
    fallback that run_react_episode has — scoring is now identical across
    all 82 combos)

Only pandas is imported at module level, so merge/gate callers never pull in
torch. Episode-loop functions import src.* lazily inside the function body.
"""

import os
from pathlib import Path

import pandas as pd

COMBO_COLS = ["policy", "slm_name", "dataset"]


# ---------------------------------------------------------------------------
# Paths and data
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def results_dir() -> Path:
    d = Path(os.environ.get("SLM_RESULTS", repo_root() / "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    return Path(os.environ.get("SLM_DATA", repo_root() / "data"))


def load_all_datasets() -> dict[str, list[dict]]:
    from src.datasets import load_samples  # lazy import
    out = {}
    for name, fname in [("hotpotqa", "hotpotqa.json"), ("musique", "musique.json")]:
        p = data_dir() / fname
        if not p.exists():
            raise SystemExit(f"[data] missing {p} — run step1_data.py first")
        samples = load_samples(p)
        ids = [s["id"] for s in samples]
        assert len(ids) == len(set(ids)), f"[data] duplicate question ids in {p}"
        assert {"id", "question", "answer", "passages"} <= set(samples[0]), \
            f"[data] schema drift in {p}: keys={sorted(samples[0])}"
        out[name] = samples
    return out


# ---------------------------------------------------------------------------
# Resume with partial-combo dedupe
# ---------------------------------------------------------------------------

def load_done_keys(csv_path: Path, expected_n: dict[str, int]):
    """
    Returns (rows, done_keys). Rows belonging to COMPLETE combos are kept and
    their keys returned; rows from partial combos are dropped (the combo is
    re-run wholesale), so resume never duplicates.
    """
    if not csv_path.exists():
        return [], set()
    df = pd.read_csv(csv_path)
    if df.empty:
        return [], set()
    keep_mask = pd.Series(False, index=df.index)
    done_keys = set()
    for (policy, slm, ds), grp in df.groupby(COMBO_COLS):
        if len(grp) >= expected_n.get(ds, 0):
            done_keys.add((policy, slm, ds))
            keep_mask.loc[grp.index] = True
        else:
            print(f"  [resume] dropping {len(grp)} partial rows for "
                  f"({policy}, {slm}, {ds}) — combo re-runs from scratch")
    kept = df[keep_mask]
    return kept.to_dict("records"), done_keys


def shard_path(base_name: str, shard: str | None) -> Path:
    """Per-shard CSV path: results/<base>.gpu0.csv etc. No shard -> unsuffixed."""
    if shard:
        return results_dir() / f"{base_name}.{shard}.csv"
    return results_dir() / f"{base_name}.csv"


def checkpoint(rows: list[dict], path: Path) -> None:
    """Atomic write: a kill mid-write cannot tear the CSV."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_shards(base_name: str, shards: list[str], expected_n: dict[str, int],
                 out_name: str | None = None,
                 dup_extra_cols: list[str] | None = None) -> pd.DataFrame:
    """dup_extra_cols: extra uniqueness key columns for row types with more
    than one row per question — step-level CSVs must pass ["step_idx"],
    otherwise every multi-step episode reads as duplicate rows."""
    frames = []
    for sh in shards:
        p = shard_path(base_name, sh)
        if not p.exists():
            raise SystemExit(f"[merge] missing shard file {p}")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    dup_key = COMBO_COLS + ["question_id"] + list(dup_extra_cols or [])
    dup = df.duplicated(subset=dup_key)
    assert not dup.any(), f"[merge] {int(dup.sum())} duplicate rows in {base_name}"

    # keep only complete combos (same rule as resume): partial rows would be
    # re-run anyway, and downstream analysis must never see them
    keep = pd.Series(False, index=df.index)
    dropped = 0
    for (policy, slm, ds), grp in df.groupby(COMBO_COLS):
        if len(grp) >= expected_n.get(ds, 0):
            keep.loc[grp.index] = True
        else:
            dropped += len(grp)
            print(f"  [merge] dropping partial combo ({policy}, {slm}, {ds}): "
                  f"{len(grp)} rows")
    df = df[keep].reset_index(drop=True)

    out = results_dir() / f"{out_name or base_name}.csv"
    df.to_csv(out, index=False)
    print(f"[merge] {len(df)} rows -> {out} (dropped {dropped} partial)")
    for (policy, slm, ds), grp in df.groupby(COMBO_COLS):
        print(f"    {policy:28s} {slm:18s} {ds:9s} n={len(grp):4d} OK")
    return df


# ---------------------------------------------------------------------------
# Episode loops (shared by step2/3/4)
# ---------------------------------------------------------------------------

def run_episode_loop(samples, ds_name, policy_id, slm_cfg, slm_model, slm_tok,
                     llm_cfg, llm_model, llm_tok, router,
                     kept_rows, out_path, checkpoint_every=20, max_steps=8,
                     kept_step_rows=None, steps_path=None):
    """
    Router-policy loop (baselines, conf/entropy/step gates) — mirrors the
    step2/step3 notebook cells with atomic periodic checkpoints.
    kept_rows: rows already complete from prior runs (written through so the
    CSV stays whole). If steps_path is given, step-level rows (logit features
    for the learned router) are checkpointed alongside. Returns
    (episode_rows, step_rows) for THIS combo only.
    """
    from src.react_env import run_react_episode, episode_to_row, steps_to_rows

    total = len(samples)
    ep_rows, step_rows = [], []
    for i, sample in enumerate(samples):
        ep = run_react_episode(
            sample, slm_model, slm_tok, slm_cfg,
            llm_model, llm_tok, llm_cfg,
            router, policy_id, max_steps=max_steps,
        )
        ep_rows.append(episode_to_row(ep))
        step_rows.extend(steps_to_rows(ep))
        if (i + 1) % checkpoint_every == 0:
            checkpoint(kept_rows + ep_rows, out_path)
            if steps_path is not None:
                checkpoint((kept_step_rows or []) + step_rows, steps_path)
            print(f"    [{i + 1}/{total}]")
    checkpoint(kept_rows + ep_rows, out_path)
    if steps_path is not None:
        checkpoint((kept_step_rows or []) + step_rows, steps_path)
    sr = sum(r["success"] for r in ep_rows) / max(len(ep_rows), 1)
    lr = sum(r["llm_call_ratio"] for r in ep_rows) / max(len(ep_rows), 1)
    print(f"    -> {policy_id} | {ds_name}: SR={sr:.3f} LLM_ratio={lr:.3f}")
    return ep_rows, step_rows


def run_inline_loop(samples, ds_name, policy_id, slm_label, generate_fn,
                    kept_rows, out_path, checkpoint_every=20, max_steps=8):
    """
    Custom-generation loop for cascade and learned policies (the notebook's
    inline cells). generate_fn(prompt_messages) -> (response_text,
    used_llm: bool). Applies the SAME scratchpad answer fallback as
    run_react_episode so scoring is uniform across policies.
    """
    from src.inference import parse_react_action, parse_final_answer
    from src.react_env import (
        build_react_prompt, _execute_search, _is_finish_action,
        _extract_finish_answer, _exact_match,
    )
    from src.datasets import build_bm25_index

    total = len(samples)
    ep_rows = []
    for i, sample in enumerate(samples):
        question, gold, passages = sample["question"], sample["answer"], sample["passages"]
        bm25 = build_bm25_index(passages)
        scratchpad, predicted = "", ""
        n_llm = n_slm = 0
        for step_idx in range(max_steps):
            prompt = build_react_prompt(question, scratchpad)
            resp, used_llm = generate_fn(prompt)
            n_llm += int(used_llm)
            n_slm += int(not used_llm)
            parsed = parse_react_action(resp)
            if _is_finish_action(parsed["action"], parsed["action_input"], resp):
                predicted = _extract_finish_answer(parsed["action"], parsed["action_input"], resp)
                break
            obs = _execute_search(parsed["action_input"] or parsed["thought"], bm25, passages)
            scratchpad += (f"\nThought: {parsed['thought']}\n"
                           f"Action: {parsed['action']}[{parsed['action_input']}]\n"
                           f"Observation: {obs}\n")
        if not predicted:  # uniform fallback (missing in the notebook inline cells)
            predicted = parse_final_answer(scratchpad) or ""
        success = _exact_match(predicted, gold)
        n_total = n_llm + n_slm
        ep_rows.append({
            "question_id": sample["id"], "dataset": ds_name, "policy": policy_id,
            "slm_name": slm_label, "success": success, "predicted": predicted,
            "gold_answer": gold, "n_steps": step_idx + 1,
            "n_llm_calls": n_llm, "n_slm_calls": n_slm,
            "llm_call_ratio": n_llm / n_total if n_total else 0.0,
        })
        if (i + 1) % checkpoint_every == 0:
            checkpoint(kept_rows + ep_rows, out_path)
            print(f"    [{i + 1}/{total}]")
    checkpoint(kept_rows + ep_rows, out_path)
    sr = sum(r["success"] for r in ep_rows) / max(len(ep_rows), 1)
    lr = sum(r["llm_call_ratio"] for r in ep_rows) / max(len(ep_rows), 1)
    print(f"    -> {policy_id} | {ds_name}: SR={sr:.3f} LLM_ratio={lr:.3f}")
    return ep_rows
