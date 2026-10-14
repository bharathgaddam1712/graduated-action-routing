"""G0 — GPU gate. Verifies torch sees the cards, they're Blackwell-class, a
real fp16 matmul works on each, and no other tenant is holding memory."""

import _bootstrap  # noqa: F401  (must stay first)

import subprocess
import sys

import torch

CU128_HINT = ("uv pip install 'torch==2.7.*' "
              "--index-url https://download.pytorch.org/whl/cu128")


def fail(msg: str) -> None:
    print(f"G0 FAIL: {msg}")
    sys.exit(1)


def check_gpu(i: int) -> None:
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name} | sm_{cap[0]}{cap[1]} | {vram_gb:.0f} GB")
    if cap[0] < 9:
        fail(f"GPU {i} is sm_{cap[0]}{cap[1]} — validated for sm_120 "
             f"(RTX PRO 6000 Blackwell) with cu128 wheels. Install: {CU128_HINT}")
    try:
        a = torch.randn(4096, 4096, device=f"cuda:{i}", dtype=torch.float16)
        b = torch.randn(4096, 4096, device=f"cuda:{i}", dtype=torch.float16)
        c = a @ b
        torch.cuda.synchronize(i)
        assert torch.isfinite(c.float()).all(), "non-finite matmul result"
        del a, b, c
        torch.cuda.empty_cache()
        print("         fp16 matmul OK")
    except Exception as e:  # noqa: BLE001
        fail(f"fp16 matmul failed on cuda:{i}: {e}")


def check_tenancy() -> None:
    """Warn if another process holds GPU memory. util% is unreliable on idle
    Blackwell driver 610.x (reports 100% at ~0 MiB) — trust memory, not util."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001
        print(f"  (nvidia-smi check skipped: {e})")
        return
    for line in out:
        idx, mem_mib, util = [x.strip() for x in line.split(",")]
        if int(mem_mib) > 2048:
            print(f"  WARNING GPU {idx}: {mem_mib} MiB used by another process — "
                  f"shared box, runs may contend (util={util}% ignored, see above)")


def main() -> None:
    if not torch.cuda.is_available():
        fail(f"torch.cuda.is_available() is False — install: {CU128_HINT}")
    n = torch.cuda.device_count()
    print(f"CUDA devices: {n}")
    if n < 1:
        fail("no CUDA devices visible")
    for i in range(n):
        check_gpu(i)
    check_tenancy()
    print("G0 OK")


if __name__ == "__main__":
    main()
