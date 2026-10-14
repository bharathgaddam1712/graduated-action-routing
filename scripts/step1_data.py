"""Step 1 — download HotpotQA(300) + MuSiQue(200), verify BM25, save JSONs.
CPU-only. Asserts counts/schema before any GPU hour is spent downstream."""

import _bootstrap  # noqa: F401  (must stay first)

import statistics
import sys

import datasets as hf_datasets


def fail(msg: str) -> None:
    print(f"STEP1 FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    import common_run as cr

    # Skip the download entirely if both JSONs already exist with the right
    # counts — relaunches (and this box's flaky network) make this the norm.
    ddir = cr.data_dir()
    hp_path, mu_path = ddir / "hotpotqa.json", ddir / "musique.json"
    if hp_path.exists() and mu_path.exists():
        import json as _json
        hp = _json.loads(hp_path.read_text())
        mu = _json.loads(mu_path.read_text())
        if len(hp) == 300 and len(mu) == 200:
            print(f"[step1] {hp_path} ({len(hp)}) + {mu_path} ({len(mu)}) "
                  f"already present — skipping download")
            print("STEP1 OK")
            return
        print("[step1] existing JSONs have wrong counts — re-downloading")

    # trust_remote_code loaders were removed in datasets 3.x — pin is in
    # requirements-server.txt; this assert catches a stray manual upgrade.
    major = int(hf_datasets.__version__.split(".")[0])
    if major >= 3:
        fail(f"datasets=={hf_datasets.__version__} breaks script datasets "
             f"(hotpot_qa needs trust_remote_code). Install: uv pip install 'datasets==2.21.*'")

    from src.datasets import load_hotpotqa, load_musique, save_samples
    from src.datasets import build_bm25_index, retrieve
    import common_run as cr

    hotpotqa = load_hotpotqa(n=300)
    musique = load_musique(n=200)

    if len(hotpotqa) != 300:
        fail(f"hotpotqa count {len(hotpotqa)} != 300")
    if len(musique) != 200:
        fail(f"musique count {len(musique)} != 200")
    for name, samples in [("hotpotqa", hotpotqa), ("musique", musique)]:
        ids = [s["id"] for s in samples]
        if len(ids) != len(set(ids)):
            fail(f"duplicate ids in {name}")
        for s in samples[:10]:
            if not (s["question"] and s["answer"] and s["passages"]):
                fail(f"empty field in {name} sample {s['id']}")

    ddir = cr.data_dir()
    ddir.mkdir(parents=True, exist_ok=True)
    save_samples(hotpotqa, ddir / "hotpotqa.json")
    save_samples(musique, ddir / "musique.json")

    print("\nBM25 spot check (first 3 hotpotqa):")
    for sample in hotpotqa[:3]:
        index = build_bm25_index(sample["passages"])
        result = retrieve(sample["question"], index, sample["passages"], top_k=1)
        print(f"  Q: {sample['question'][:80]}")
        print(f"  A: {sample['answer']}  | retrieved {len(result)} chars")

    hp = [len(s["passages"]) for s in hotpotqa]
    mu = [len(s["passages"]) for s in musique]
    print(f"\nHotpotQA avg passages/q: {statistics.mean(hp):.1f}")
    print(f"MuSiQue  avg passages/q: {statistics.mean(mu):.1f}")
    print("STEP1 OK")


if __name__ == "__main__":
    main()
