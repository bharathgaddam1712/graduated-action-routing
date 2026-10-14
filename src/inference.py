"""
Inference utilities for Paper B.
Copied from light_paper/src/inference.py with one addition:
- generate_single_with_logits(): single-item generation that also returns
  raw logit features needed by the learned router (top-1 prob, entropy,
  top-1 minus top-2 margin, mean top-5 prob).
"""

import re
import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Chat template formatting
# ---------------------------------------------------------------------------

def format_messages(messages: list[dict], tokenizer: PreTrainedTokenizerBase) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"


# ---------------------------------------------------------------------------
# Batched generation (used for baselines)
# ---------------------------------------------------------------------------

def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[list[dict]],
    max_new_tokens: int = 256,
    batch_size: int = 4,
    temperature: float = 0.0,
) -> list[str]:
    all_responses = []
    device = next(model.parameters()).device

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        formatted = [format_messages(m, tokenizer) for m in batch]
        enc = tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for seq in out:
            decoded = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
            all_responses.append(decoded)

        if i % (batch_size * 5) == 0:
            print(f"  [inference] {i + len(batch)}/{len(prompts)}")

    return all_responses


# ---------------------------------------------------------------------------
# Single-item generation with full logit features (used in routing loop)
# ---------------------------------------------------------------------------

def generate_single_with_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    max_new_tokens: int = 256,
) -> dict:
    """
    Generate one response and return text + logit-derived routing features.

    Returns:
        response:         decoded string
        first_token_entropy:    H(P(first token))
        mean_entropy:           mean token entropy across all generated tokens
        top1_prob:              P(argmax) of first token
        margin_top1_top2:       P(top1) - P(top2) of first token
        mean_top5_prob:         mean of top-5 probabilities of first token
    """
    device = next(model.parameters()).device
    formatted = format_messages(messages, tokenizer)
    enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = enc["input_ids"].shape[1]
    response = tokenizer.decode(out.sequences[0][input_len:], skip_special_tokens=True).strip()

    # Logit features from first generated token
    if out.scores:
        first_logits = out.scores[0][0]            # (vocab,)
        probs = torch.softmax(first_logits, dim=-1).cpu().float().numpy()
        sorted_probs = np.sort(probs)[::-1]

        entropy = float(-(probs * np.log(probs + 1e-9)).sum())
        top1_prob = float(sorted_probs[0])
        margin    = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        mean_top5 = float(sorted_probs[:5].mean())

        # All-token mean entropy + window_conf (AdaptEvolve: mean of bottom 20% log-probs)
        all_entropies = []
        all_logprobs  = []
        for score in out.scores:
            p = torch.softmax(score[0], dim=-1).cpu().float().numpy()
            all_entropies.append(float(-(p * np.log(p + 1e-9)).sum()))
            all_logprobs.append(float(np.log(p.max() + 1e-9)))
        mean_entropy = float(np.mean(all_entropies))
        # bottom 20% of per-token log-probs → high value = uncertain (AdaptEvolve BWC signal)
        k = max(1, len(all_logprobs) // 5)
        window_conf  = float(np.mean(sorted(all_logprobs)[:k]))
    else:
        entropy = mean_entropy = top1_prob = margin = mean_top5 = window_conf = 0.0

    return {
        "response":             response,
        "first_token_entropy":  entropy,
        "mean_entropy":         mean_entropy,
        "top1_prob":            top1_prob,
        "margin_top1_top2":     margin,
        "mean_top5_prob":       mean_top5,
        "window_conf":          window_conf,   # AdaptEvolve BWC signal (negative log-prob, lower = more uncertain)
    }


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def parse_final_answer(response: str) -> str | None:
    """
    Extract final answer from a ReAct response that ends with:
      Final Answer: <answer>
    Falls back to the last non-empty line.
    """
    match = re.search(r"[Ff]inal [Aa]nswer[:\s]+(.+)", response)
    if match:
        return match.group(1).strip().rstrip(".")
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1] if lines else None


def parse_react_action(response: str) -> dict:
    """
    Extract Thought / Action / Action Input from a ReAct step response.
    Returns dict with keys: thought, action, action_input.
    Missing fields default to empty string.
    """
    thought = ""
    action  = ""
    action_input = ""

    t = re.search(r"[Tt]hought[:\s]+(.+?)(?=\n[A-Z]|\Z)", response, re.DOTALL)
    a = re.search(r"[Aa]ction[:\s]+(.+?)(?=\n[A-Z]|\Z)", response)
    i = re.search(r"[Aa]ction [Ii]nput[:\s]+(.+?)(?=\n[A-Z]|\Z)", response, re.DOTALL)

    if t: thought = t.group(1).strip()
    if a: action  = a.group(1).strip().split("\n")[0]
    if i: action_input = i.group(1).strip().split("\n")[0]

    return {"thought": thought, "action": action, "action_input": action_input}


def parse_confidence(response: str) -> float | None:
    """Extract verbalized confidence 0-100, normalize to 0-1. Returns None if absent."""
    m = re.search(r"[Cc]onfidence[:\s]+(\d+)", response)
    if m:
        return max(0.0, min(100.0, int(m.group(1)))) / 100.0
    nums = re.findall(r"\b(\d{1,3})\b", response)
    if nums:
        v = int(nums[-1])
        if 0 <= v <= 100:
            return v / 100.0
    return None
