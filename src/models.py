"""
Model registry for Paper B (SLM-Agents).
4 models: 3 SLM levels + 1 LLM oracle. All local, no paid API.
Adapted from light_paper/src/models.py — same load_model() signature.

Server port: fp16 everywhere. The 4-bit NF4 path was a Kaggle-T4 VRAM
constraint; on 96GB cards the full 1.5B+3B+8B cascade is ~25GB in fp16, and
dropping bitsandbytes removes the sm_120 (Blackwell) compat risk.
"""

from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


@dataclass
class ModelConfig:
    name: str
    hf_id: str
    size_b: float
    load_in_4bit: bool
    level: int          # 1=cheapest SLM, 2=mid SLM, 3=LLM oracle


MODEL_REGISTRY: list[ModelConfig] = [
    ModelConfig("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct",               1.5, False, 1),
    ModelConfig("Qwen2.5-3B",   "Qwen/Qwen2.5-3B-Instruct",                 3.0, False, 2),
    ModelConfig("Phi-3-mini",   "microsoft/Phi-3-mini-4k-instruct",          3.8, False, 2),
    ModelConfig("Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B-Instruct",    8.0, False, 3),
]

SLM_MODELS = [m for m in MODEL_REGISTRY if m.level < 3]   # routing candidates
LLM_MODEL  = next(m for m in MODEL_REGISTRY if m.level == 3)

MODEL_NAMES = [m.name for m in MODEL_REGISTRY]
SIZE_MAP    = {m.name: m.size_b for m in MODEL_REGISTRY}


def load_model(config: ModelConfig, device: str = "auto"):
    """Load model + tokenizer. Returns (model, tokenizer)."""
    print(f"[load] {config.name} ({config.hf_id})")

    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.hf_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    model.eval()
    return model, tokenizer


def get_model_by_name(name: str) -> ModelConfig:
    for m in MODEL_REGISTRY:
        if m.name == name:
            return m
    raise ValueError(f"Unknown model: {name}. Valid: {MODEL_NAMES}")
