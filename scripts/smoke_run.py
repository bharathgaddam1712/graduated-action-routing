"""G2 — tiny end-to-end run: 5 episodes each of always_slm_1b and always_llm
on HotpotQA. Verifies the whole loop (prompt -> generate -> route -> retrieve
-> parse -> score -> CSV) before real hours are spent. ~10 minutes."""

import _bootstrap  # noqa: F401  (must stay first)

import gc
import sys

import torch

import common_run as cr

N_SMOKE = 5


def fail(msg: str) -> None:
    print(f"G2 FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    from src.models import load_model, get_model_by_name
    from src.routing import AlwaysSLM, AlwaysLLM

    all_data = cr.load_all_datasets()
    samples = all_data["hotpotqa"][:N_SMOKE]

    smoke_dir = cr.results_dir() / "smoke"
    smoke_dir.mkdir(exist_ok=True)
    out_path = smoke_dir / "episodes.csv"

    slm_cfg = get_model_by_name("Qwen2.5-1.5B")
    llm_cfg = get_model_by_name("Llama-3.1-8B")

    rows = []

    # always_slm_1b: SLM only (llm dummy never used)
    slm_model, slm_tok = load_model(slm_cfg)
    ep_rows, _ = cr.run_episode_loop(
        samples, "hotpotqa", "always_slm_1b", slm_cfg, slm_model, slm_tok,
        slm_cfg, slm_model, slm_tok, AlwaysSLM(),
        kept_rows=[], out_path=out_path, checkpoint_every=N_SMOKE,
    )
    rows += ep_rows
    del slm_model, slm_tok
    gc.collect()
    torch.cuda.empty_cache()

    # always_llm: 8B plays both roles
    llm_model, llm_tok = load_model(llm_cfg)
    ep_rows, _ = cr.run_episode_loop(
        samples, "hotpotqa", "always_llm", llm_cfg, llm_model, llm_tok,
        llm_cfg, llm_model, llm_tok, AlwaysLLM(),
        kept_rows=rows, out_path=out_path, checkpoint_every=N_SMOKE,
    )
    rows += ep_rows
    del llm_model, llm_tok
    gc.collect()
    torch.cuda.empty_cache()

    if not out_path.exists():
        fail("smoke CSV was not written")
    slm_rows = [r for r in rows if r["policy"] == "always_slm_1b"]
    llm_rows = [r for r in rows if r["policy"] == "always_llm"]
    if len(slm_rows) != N_SMOKE or len(llm_rows) != N_SMOKE:
        fail(f"row counts wrong: slm={len(slm_rows)} llm={len(llm_rows)}")

    slm_ratio = sum(r["llm_call_ratio"] for r in slm_rows) / N_SMOKE
    llm_ratio = sum(r["llm_call_ratio"] for r in llm_rows) / N_SMOKE
    print(f"smoke llm_call_ratio: always_slm={slm_ratio:.2f} always_llm={llm_ratio:.2f}")
    if slm_ratio != 0.0:
        fail(f"always_slm used the LLM (ratio={slm_ratio})")
    if llm_ratio != 1.0:
        fail(f"always_llm did not always escalate (ratio={llm_ratio})")

    empties = sum(1 for r in rows if not str(r["predicted"]).strip() and r["n_steps"] == 0)
    if empties:
        fail(f"{empties} episodes produced no steps at all")

    print(f"G2 OK — {len(rows)} smoke episodes in {out_path}")


if __name__ == "__main__":
    main()
