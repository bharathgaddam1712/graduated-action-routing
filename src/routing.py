"""
Routing policies for Paper B.

7 policies, all sharing the same Router base interface:
    router.decide(response, entropy, confidence, step_idx) -> bool
    True = escalate to LLM, False = keep SLM

Policies:
  1. AlwaysSLM          — never escalate (baseline)
  2. AlwaysLLM          — always escalate (oracle)
  3. ConfidenceGateRouter(threshold) — escalate if verbalized confidence < T
  4. EntropyGateRouter(threshold)    — escalate if first_token_entropy > T
  5. StepCountRouter(k)              — escalate after step k
  6. ThreeLevelCascadeRouter(t1,t2)  — 1.5B→3B→8B double-threshold
  7. LearnedRouter(clf)              — sklearn classifier on logit features
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Router:
    needs_confidence: bool = False   # set True if policy needs verbalized conf

    def decide(self, response: str, entropy: float, confidence: float | None, step_idx: int) -> bool:
        """Return True to escalate to LLM, False to keep SLM."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Policy 1 & 2: Trivial baselines
# ---------------------------------------------------------------------------

class AlwaysSLM(Router):
    def decide(self, response, entropy, confidence, step_idx) -> bool:
        return False


class AlwaysLLM(Router):
    def decide(self, response, entropy, confidence, step_idx) -> bool:
        return True


# ---------------------------------------------------------------------------
# Policy 3: Verbalized confidence gate
# ---------------------------------------------------------------------------

class ConfidenceGateRouter(Router):
    needs_confidence = True

    def __init__(self, threshold: float):
        """threshold: escalate if confidence < threshold (both in [0,1])."""
        self.threshold = threshold

    def decide(self, response, entropy, confidence, step_idx) -> bool:
        if confidence is None:
            return True   # missing confidence → escalate (safe default)
        return confidence < self.threshold

    def __repr__(self):
        return f"ConfidenceGate(T={self.threshold:.2f})"


# ---------------------------------------------------------------------------
# Policy 4: Token entropy gate
# ---------------------------------------------------------------------------

class EntropyGateRouter(Router):
    def __init__(self, threshold: float):
        """threshold: escalate if first_token_entropy > threshold."""
        self.threshold = threshold

    def decide(self, response, entropy, confidence, step_idx) -> bool:
        return entropy > self.threshold

    def __repr__(self):
        return f"EntropyGate(T={self.threshold:.2f})"


# ---------------------------------------------------------------------------
# Policy 5: Step-count gate
# ---------------------------------------------------------------------------

class StepCountRouter(Router):
    def __init__(self, k: int):
        """Escalate for all steps with index >= k."""
        self.k = k

    def decide(self, response, entropy, confidence, step_idx) -> bool:
        return step_idx >= self.k

    def __repr__(self):
        return f"StepCount(k={self.k})"


# ---------------------------------------------------------------------------
# Policy 6: Three-level cascade (1.5B → 3B → 8B)
# Used differently: the run loop calls this router twice per step.
# First call: decides 1.5B → 3B (uses t1).
# Second call: decides 3B → 8B (uses t2).
# The notebook handles loading both SLMs and passing the right one.
# ---------------------------------------------------------------------------

class ThreeLevelCascadeRouter:
    """
    Two-threshold entropy cascade.
    level 1→2: entropy > t1
    level 2→3: entropy > t2 (t2 > t1 typically)

    Usage in the run loop:
        decision_12 = router.decide_level1(entropy)   # 1.5B or promote to 3B?
        if decision_12:
            re-run with 3B, get new entropy
            decision_23 = router.decide_level2(new_entropy)  # 3B or promote to 8B?
    """
    def __init__(self, t1: float, t2: float):
        self.t1 = t1   # 1.5B → 3B threshold
        self.t2 = t2   # 3B → 8B threshold

    def decide_level1(self, entropy: float) -> bool:
        return entropy > self.t1

    def decide_level2(self, entropy: float) -> bool:
        return entropy > self.t2

    def __repr__(self):
        return f"ThreeLevelCascade(t1={self.t1:.2f},t2={self.t2:.2f})"


# ---------------------------------------------------------------------------
# Policy 7: Learned router (logistic regression on SLM logit features)
# ---------------------------------------------------------------------------

LOGIT_FEATURES = [
    "first_token_entropy",
    "mean_entropy",
    "top1_prob",
    "margin_top1_top2",
    "mean_top5_prob",
    "window_conf",       # AdaptEvolve BWC signal — bottom 20% log-prob mean
]

# For step_idx ablation in step4: include step position as a feature
LOGIT_FEATURES_WITH_STEP = LOGIT_FEATURES + ["step_idx"]


class LearnedRouter(Router):
    def __init__(self, clf, features: list[str] | None = None):
        """clf: trained sklearn Pipeline (StandardScaler + LogisticRegression).
        features: the exact feature list clf was trained on — inference must
        use the same columns in the same order (defaults to LOGIT_FEATURES)."""
        self.clf = clf
        self.features = list(features) if features else list(LOGIT_FEATURES)
        self._threshold = 0.5   # probability threshold for escalation

    def set_threshold(self, t: float):
        self._threshold = t

    def decide_from_features(self, features: dict) -> bool:
        x = np.array([[features.get(f, 0.0) for f in self.features]])
        prob_escalate = self.clf.predict_proba(x)[0][1]
        return prob_escalate >= self._threshold

    def decide(self, response, entropy, confidence, step_idx) -> bool:
        # This is called with scalar entropy — use entropy as a fallback signal
        # The full feature vector version is used in sweep_thresholds_learned
        return entropy > 1.5   # fallback; overridden by decide_from_features

    def __repr__(self):
        return f"LearnedRouter(T={self._threshold:.2f})"


def train_learned_router(steps_df: pd.DataFrame, verbose: bool = True) -> tuple:
    """
    Train a logistic regression router on per-step logit features.

    Labels: 1 = this step needed LLM (SLM was wrong), 0 = SLM was fine.
    Proxy label: episode_success (episode-level EM used as step-level signal).
    Better labels come from the baseline run where we know which steps the SLM got right.

    Returns (LearnedRouter, auc_score, feature_importances_dict)
    """
    # Train only on features actually present in the CSV. baseline_steps.csv
    # predates window_conf being threaded through StepResult, so it has 5 of
    # the 6 LOGIT_FEATURES — intersect instead of KeyError-ing.
    feats = [f for f in LOGIT_FEATURES if f in steps_df.columns]
    missing = [f for f in LOGIT_FEATURES if f not in steps_df.columns]
    if missing:
        print(f"[routing] features not in steps CSV, excluded from training: {missing}")
    df = steps_df.dropna(subset=feats)

    # Label: step came from an episode where SLM failed → SLM needed help
    # Steps from always_slm baseline where episode_success=False are "hard" steps
    X = df[feats].values
    y = (~df["episode_success"]).astype(int).values   # 1 = SLM failed, should escalate

    if len(np.unique(y)) < 2:
        print("[routing] Warning: only one class in training labels — LearnedRouter may not work")

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
    ])
    clf.fit(X, y)

    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1]) if len(np.unique(y)) > 1 else 0.5

    # Feature importances (LR coefficients after scaling)
    coefs = clf.named_steps["lr"].coef_[0]
    importances = dict(zip(feats, coefs))

    if verbose:
        print(f"[routing] LearnedRouter trained: AUC={auc:.3f}, n_train={len(df)}")
        for feat, coef in sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feat:30s}: {coef:+.3f}")

    return LearnedRouter(clf, feats), float(auc), importances


# ---------------------------------------------------------------------------
# Threshold sweep helpers
# ---------------------------------------------------------------------------

CONF_THRESHOLDS    = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ENTROPY_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
STEP_KS            = [1, 2, 3, 4, 5, 6]
LEARNED_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

CASCADE_GRID = [
    (0.5, 1.5), (0.5, 2.0), (0.5, 2.5),
    (1.0, 2.0), (1.0, 2.5), (1.0, 3.0),
    (1.5, 2.5), (1.5, 3.0),
]


def make_routers_for_sweep() -> dict[str, list[Router]]:
    """Return all (policy_id, router) pairs for the full threshold sweep."""
    routers = {
        "always_slm": [AlwaysSLM()],
        "always_llm": [AlwaysLLM()],
        "conf_gate":  [ConfidenceGateRouter(t) for t in CONF_THRESHOLDS],
        "entropy_gate": [EntropyGateRouter(t) for t in ENTROPY_THRESHOLDS],
        "step_gate":  [StepCountRouter(k) for k in STEP_KS],
    }
    return routers
