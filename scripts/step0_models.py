"""G1 — model gate. Downloads all 4 models fp16 into HF_HOME and runs a
20-token generation on each. The Llama-3.1-8B load doubles as the HF_TOKEN /
license gate: it fails here, before any GPU hours are spent."""

import _bootstrap  # noqa: F401  (must stay first)

import gc
import os
import sys

import torch


def fail(msg: str) -> None:
    print(f"G1 FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    from src.models import MODEL_REGISTRY, load_model
    from src.inference import generate_single_with_logits

    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN is not set. Qwen/Phi download fine without it, "
              "but meta-llama/Meta-Llama-3.1-8B-Instruct is GATED.")
        print("  1. Create a token: https://huggingface.co/settings/tokens")
        print("  2. Accept the license: "
              "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
        print("  3. Put HF_TOKEN=hf_... in .env and re-source env.sh")

    for cfg in MODEL_REGISTRY:
        print(f"\n--- {cfg.name} ({cfg.hf_id}) ---")
        try:
            model, tok = load_model(cfg)
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "gated" in msg or "401" in msg or "403" in msg:
                fail(f"{cfg.hf_id} is gated and your token lacks access. "
                     f"Accept the license on the model page, then retry. ({e})")
            fail(f"load failed for {cfg.hf_id}: {e}")

        out = generate_single_with_logits(
            model, tok,
            [{"role": "user", "content": "Reply with the single word: ready"}],
            max_new_tokens=20,
        )
        text = out["response"] or ""
        ent = out["first_token_entropy"]
        print(f"  gen: {text[:60]!r} | entropy={ent:.3f} top1={out['top1_prob']:.3f}")
        if not text.strip():
            fail(f"{cfg.name} generated empty text — check chat template/dtype")
        if ent != ent:  # NaN
            fail(f"{cfg.name} produced NaN entropy — fp16 instability, do not proceed")

        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
        print("  OK, unloaded")

    print("\nG1 OK — all models cached in", os.environ.get("HF_HOME", "<default>"))


if __name__ == "__main__":
    main()
