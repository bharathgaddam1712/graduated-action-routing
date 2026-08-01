"""
ReAct environment for Paper B.

Implements the full ReAct loop:
  Thought → Action → Action Input → [tool call] → Observation → repeat
  ... → Final Answer

The loop is router-aware: at each Thought/Action step, the router decides
whether to use the SLM or escalate to the LLM.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from src.inference import (
    format_messages,
    generate_single_with_logits,
    parse_react_action,
    parse_final_answer,
    parse_confidence,
)
from src.datasets import build_bm25_index, retrieve


# ---------------------------------------------------------------------------
# ReAct prompt templates
# ---------------------------------------------------------------------------

REACT_SYSTEM = """You are a reasoning agent that solves multi-hop questions by alternating between thinking and using tools.

Available tools:
- Search[query]: Search for relevant information. Returns the most relevant passages.
- Finish[answer]: Provide your final answer and stop.

Format each step exactly as:
Thought: <your reasoning>
Action: Search[<query>] or Finish[<answer>]

Do not include anything else after an Action line."""


def build_react_prompt(question: str, scratchpad: str) -> list[dict]:
    """
    Build the chat messages list for the next ReAct step.
    scratchpad accumulates all previous Thought/Action/Observation turns.
    """
    user_content = f"Question: {question}"
    if scratchpad:
        user_content += f"\n\n{scratchpad.strip()}"

    return [
        {"role": "system",    "content": REACT_SYSTEM},
        {"role": "user",      "content": user_content},
    ]


def build_confidence_elicitation_prompt(question: str, scratchpad: str, proposed_action: str) -> list[dict]:
    """
    After SLM proposes an action, ask it to rate its own confidence.
    Used by the verbalized confidence routing policy.
    """
    user_content = (
        f"Question: {question}\n\n{scratchpad.strip()}\n\n"
        f"You proposed:\n{proposed_action}\n\n"
        "How confident are you that this action is correct and will help answer the question? "
        "Rate 0–100 where 0=completely uncertain, 100=certain. "
        "Respond with exactly: Confidence: <number>"
    )
    return [
        {"role": "system", "content": "You are a calibrated confidence estimator."},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Step result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step_idx:      int
    model_used:    str      # "slm" or "llm"
    slm_name:      str      # name of the SLM model config
    thought:       str
    action:        str
    action_input:  str
    observation:   str
    is_terminal:   bool     # True if this step calls Finish[...]
    # Routing features (from SLM inference)
    first_token_entropy:  float = 0.0
    mean_entropy:         float = 0.0
    top1_prob:            float = 0.0
    margin_top1_top2:     float = 0.0
    mean_top5_prob:       float = 0.0
    verbalized_conf:      float | None = None
    router_decision:      str = ""   # e.g. "conf_gate:escalated" or "entropy_gate:slm"


@dataclass
class EpisodeResult:
    question_id:  str
    question:     str
    gold_answer:  str
    dataset:      str
    policy:       str
    slm_name:     str
    success:      bool        # EM with gold_answer
    predicted:    str
    n_steps:      int
    n_llm_calls:  int
    n_slm_calls:  int
    steps:        list[StepResult] = field(default_factory=list)

    @property
    def llm_call_ratio(self) -> float:
        total = self.n_llm_calls + self.n_slm_calls
        return self.n_llm_calls / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Core ReAct loop
# ---------------------------------------------------------------------------

def run_react_episode(
    sample: dict,
    slm_model,
    slm_tokenizer,
    slm_config,
    llm_model,
    llm_tokenizer,
    llm_config,
    router,             # Router instance from routing.py
    policy_name: str,
    max_steps: int = 8,
) -> EpisodeResult:
    """
    Run one full ReAct episode with routing.

    The router is called after every SLM generation to decide whether to
    re-run that step with the LLM instead.

    Returns an EpisodeResult with full step-level logging.
    """
    question = sample["question"]
    gold     = sample["answer"]
    passages = sample["passages"]

    # Build per-question BM25 index
    bm25_index = build_bm25_index(passages)

    scratchpad  = ""
    steps: list[StepResult] = []
    n_llm = 0
    n_slm = 0
    predicted = ""

    for step_idx in range(max_steps):
        prompt = build_react_prompt(question, scratchpad)

        # ── Step A: always generate with SLM first ─────────────────────
        slm_out = generate_single_with_logits(slm_model, slm_tokenizer, prompt)
        slm_response = slm_out["response"]

        # Ask SLM for verbalized confidence (only for conf_gate policy — cheap, one extra call)
        verbalized_conf = None
        if hasattr(router, "needs_confidence") and router.needs_confidence:
            conf_prompt = build_confidence_elicitation_prompt(question, scratchpad, slm_response)
            conf_out    = generate_single_with_logits(slm_model, slm_tokenizer, conf_prompt, max_new_tokens=16)
            verbalized_conf = parse_confidence(conf_out["response"])

        # ── Step B: router decides whether to escalate ──────────────────
        escalate = router.decide(
            response    = slm_response,
            entropy     = slm_out["first_token_entropy"],
            confidence  = verbalized_conf,
            step_idx    = step_idx,
        )

        if escalate:
            # Re-run this step with LLM
            llm_out      = generate_single_with_logits(llm_model, llm_tokenizer, prompt)
            used_response = llm_out["response"]
            model_used    = "llm"
            n_llm += 1
            router_decision = f"{policy_name}:escalated"
        else:
            used_response = slm_response
            model_used    = "slm"
            n_slm += 1
            router_decision = f"{policy_name}:slm"

        # ── Step C: parse and execute ────────────────────────────────────
        parsed = parse_react_action(used_response)
        thought      = parsed["thought"]
        action       = parsed["action"]
        action_input = parsed["action_input"]

        # Check for terminal action
        is_terminal = _is_finish_action(action, action_input, used_response)

        if is_terminal:
            predicted = _extract_finish_answer(action, action_input, used_response)
            observation = ""
        else:
            # Execute Search tool
            observation = _execute_search(action_input or thought, bm25_index, passages)

        # Update scratchpad
        scratchpad += f"\nThought: {thought}\nAction: {action}[{action_input}]\n"
        if observation:
            scratchpad += f"Observation: {observation}\n"

        steps.append(StepResult(
            step_idx             = step_idx,
            model_used           = model_used,
            slm_name             = slm_config.name,
            thought              = thought,
            action               = action,
            action_input         = action_input,
            observation          = observation[:200],  # truncate for storage
            is_terminal          = is_terminal,
            first_token_entropy  = slm_out["first_token_entropy"],
            mean_entropy         = slm_out["mean_entropy"],
            top1_prob            = slm_out["top1_prob"],
            margin_top1_top2     = slm_out["margin_top1_top2"],
            mean_top5_prob       = slm_out["mean_top5_prob"],
            verbalized_conf      = verbalized_conf,
            router_decision      = router_decision,
        ))

        if is_terminal:
            break

    # If no terminal step reached, use last action_input as answer
    if not predicted:
        predicted = parse_final_answer(scratchpad) or ""

    success = _exact_match(predicted, gold)

    return EpisodeResult(
        question_id = sample["id"],
        question    = question[:200],
        gold_answer = gold,
        dataset     = sample.get("dataset", "unknown"),
        policy      = policy_name,
        slm_name    = slm_config.name,
        success     = success,
        predicted   = predicted,
        n_steps     = len(steps),
        n_llm_calls = n_llm,
        n_slm_calls = n_slm,
        steps       = steps,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_finish_action(action: str, action_input: str, response: str) -> bool:
    action_lower = action.lower()
    if "finish" in action_lower:
        return True
    if re.search(r"[Ff]inish\[", response):
        return True
    if re.search(r"[Ff]inal [Aa]nswer", response):
        return True
    return False


def _extract_finish_answer(action: str, action_input: str, response: str) -> str:
    m = re.search(r"[Ff]inish\[(.+?)\]", response)
    if m:
        return m.group(1).strip()
    m = re.search(r"[Ff]inal [Aa]nswer[:\s]+(.+)", response)
    if m:
        return m.group(1).strip().rstrip(".")
    return action_input.strip() if action_input else ""


def _execute_search(query: str, bm25_index, passages: list[str]) -> str:
    if not query.strip():
        return "No query provided."
    result = retrieve(query, bm25_index, passages, top_k=2)
    return result[:500] if result else "No relevant passages found."


def _exact_match(predicted: str, gold: str) -> bool:
    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = re.sub(r"[^\w\s]", "", s)
        return " ".join(s.split())

    return normalize(predicted) == normalize(gold)


def episode_to_row(ep: EpisodeResult) -> dict:
    """Flatten EpisodeResult to a single CSV row (no step-level detail)."""
    return {
        "question_id":   ep.question_id,
        "dataset":       ep.dataset,
        "policy":        ep.policy,
        "slm_name":      ep.slm_name,
        "success":       ep.success,
        "predicted":     ep.predicted,
        "gold_answer":   ep.gold_answer,
        "n_steps":       ep.n_steps,
        "n_llm_calls":   ep.n_llm_calls,
        "n_slm_calls":   ep.n_slm_calls,
        "llm_call_ratio": ep.llm_call_ratio,
    }


def steps_to_rows(ep: EpisodeResult) -> list[dict]:
    """Expand EpisodeResult to per-step rows for the learned router training data."""
    rows = []
    for s in ep.steps:
        rows.append({
            "question_id":          ep.question_id,
            "dataset":              ep.dataset,
            "policy":               ep.policy,
            "slm_name":             ep.slm_name,
            "step_idx":             s.step_idx,
            "model_used":           s.model_used,
            "is_terminal":          s.is_terminal,
            "first_token_entropy":  s.first_token_entropy,
            "mean_entropy":         s.mean_entropy,
            "top1_prob":            s.top1_prob,
            "margin_top1_top2":     s.margin_top1_top2,
            "mean_top5_prob":       s.mean_top5_prob,
            "verbalized_conf":      s.verbalized_conf,
            "episode_success":      ep.success,   # label for learned router
        })
    return rows
