"""
Authoritative Single Production Decision Policy for Abuse-Ring Sentinel.

Implements the frozen validation thresholds:
- Model D High Risk (r_gbm >= 0.50) -> BLOCK
- Sentinel Coordinated Ring Risk (r_ring >= 0.45) -> MANUAL_REVIEW
- Model D Moderate Risk (r_gbm >= 0.05) -> MANUAL_REVIEW
- Otherwise -> ALLOW

This module is the SINGLE source of truth for:
- API alert detail endpoints (/api/alerts, /api/alerts/{id}, /api/transaction/{id})
- API demo cases endpoint (/api/demo/cases)
- Offline business evaluation (src/evaluation/business_evaluation.py)
- Policy simulation baseline
"""

import os
import yaml
from typing import Tuple, Dict, Any

DEFAULT_D_BLOCK_THRESHOLD = 0.50
DEFAULT_D_REVIEW_THRESHOLD = 0.05
DEFAULT_SENTINEL_THRESHOLD = 0.45

def load_production_thresholds() -> Tuple[float, float, float]:
    """Load frozen production thresholds from config or use defaults."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "configs", "risk_policy.yaml"
    )
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                prod = cfg.get("production_policy", {})
                d_block = float(prod.get("model_d_block_threshold", DEFAULT_D_BLOCK_THRESHOLD))
                d_review = float(prod.get("model_d_review_threshold", DEFAULT_D_REVIEW_THRESHOLD))
                sentinel_thresh = float(prod.get("sentinel_threshold", DEFAULT_SENTINEL_THRESHOLD))
                return d_block, d_review, sentinel_thresh
        except Exception:
            pass
    return DEFAULT_D_BLOCK_THRESHOLD, DEFAULT_D_REVIEW_THRESHOLD, DEFAULT_SENTINEL_THRESHOLD

def decide_hybrid_policy(
    r_gbm: float,
    r_ring: float,
    d_block: float = None,
    d_review: float = None,
    sentinel_threshold: float = None
) -> Tuple[str, str]:
    """
    Authoritative single decision policy.
    
    Returns:
        (decision, flagged_reason)
        where decision is one of: "BLOCK", "MANUAL_REVIEW", "ALLOW"
    """
    if d_block is None or d_review is None or sentinel_threshold is None:
        cfg_block, cfg_review, cfg_sentinel = load_production_thresholds()
        d_block = d_block if d_block is not None else cfg_block
        d_review = d_review if d_review is not None else cfg_review
        sentinel_threshold = sentinel_threshold if sentinel_threshold is not None else cfg_sentinel

    if r_gbm >= d_block:
        return "BLOCK", "MODEL_D_HIGH_RISK"
    elif r_ring >= sentinel_threshold:
        return "MANUAL_REVIEW", "SENTINEL_RING_ESCALATION"
    elif r_gbm >= d_review:
        return "MANUAL_REVIEW", "MODEL_D_MODERATE_RISK"
    else:
        return "ALLOW", "NORMAL"

def decide_model_d(
    r_gbm: float,
    d_block: float = None,
    d_review: float = None
) -> Tuple[str, str]:
    """
    Decision policy for Model D individual tabular model alone.
    """
    if d_block is None or d_review is None:
        cfg_block, cfg_review, _ = load_production_thresholds()
        d_block = d_block if d_block is not None else cfg_block
        d_review = d_review if d_review is not None else cfg_review

    if r_gbm >= d_block:
        return "BLOCK", "MODEL_D_HIGH_RISK"
    elif r_gbm >= d_review:
        return "MANUAL_REVIEW", "MODEL_D_MODERATE_RISK"
    else:
        return "ALLOW", "NORMAL"
