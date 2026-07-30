"""
Gating Functions for TGAI Benchmark

Implements trust-weighted auxiliary information gating mechanisms:
- Logistic gate: w = σ(-a·D(x) + b·S(x) + c)
- Linear gate: w = clip(α·D(x) + β, 0, 1)
- Threshold gate: w = 1 if D(x) < τ, else 0

Also provides evaluation metrics:
- Expected Calibration Error (ECE)
- Area Under ROC Curve (AUC-ROC)
- Brier Score
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.stats import roc_auc_score


# =============================================================================
# GATING FUNCTIONS
# =============================================================================

def gate_logistic(divergence: np.ndarray, 
                  stakes: np.ndarray,
                  a: float = 1.0, 
                  b: float = 1.0, 
                  c: float = 0.0) -> np.ndarray:
    """
    Logistic gate: w = σ(-a·D(x) + b·S(x) + c)
    
    Args:
        divergence: Divergence statistic D(x) ∈ [0, 1]
        stakes: Stakes asymmetry S(x) ≥ 0
        a: Coefficient on divergence (typically positive, so -a makes w decrease with divergence)
        b: Coefficient on stakes
        c: Intercept term
    
    Returns:
        w ∈ [0, 1] representing trust in prior-informed track
    
    Raises:
        ValueError: If any input is NaN or infinite
    """
    # Input validation
    divergence = np.asarray(divergence)
    stakes = np.asarray(stakes)
    
    if np.any(~np.isfinite(divergence)) or np.any(~np.isfinite(stakes)):
        raise ValueError("Divergence and stakes must be finite values")
    
    if divergence.ndim != 1 or stakes.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    
    if divergence.shape != stakes.shape:
        raise ValueError("Divergence and stakes must have same shape")
    
    # Clip divergence to [0, 1] to handle floating point errors
    divergence = np.clip(divergence, 0.0, 1.0)
    
    # Compute logit with numerical stability
    z = -a * divergence + b * stakes + c
    
    # Clip z to avoid overflow in exp()
    z = np.clip(z, -500, 500)
    
    # Sigmoid with numerical stability
    w = 1.0 / (1.0 + np.exp(-z))
    
    # Ensure w stays in (0, 1) to avoid division by zero later
    w = np.clip(w, 1e-10, 1 - 1e-10)
    
    return w


def gate_linear(divergence: np.ndarray,
                stakes: np.ndarray,
                alpha_coef: float = 0.5,
                beta_coef: float = 0.5) -> np.ndarray:
    """
    Linear gate: w = clip(α·D(x) + β, 0, 1)
    
    Args:
        divergence: Divergence statistic D(x) ∈ [0, 1]
        stakes: Not used in linear gate (kept for API consistency)
        alpha_coef: Slope coefficient (should be negative for "low divergence = high trust")
        beta_coef: Intercept
    
    Returns:
        w ∈ [0, 1]
    """
    divergence = np.asarray(divergence)
    
    if np.any(~np.isfinite(divergence)):
        raise ValueError("Divergence must be finite values")
    
    # Clip divergence to [0, 1]
    divergence = np.clip(divergence, 0.0, 1.0)
    
    # Linear combination
    w = alpha_coef * divergence + beta_coef
    
    # Clip to valid range
    w = np.clip(w, 0.0, 1.0)
    
    return w


def gate_threshold(divergence: np.ndarray,
                   stakes: np.ndarray,
                   tau: float = 0.3) -> np.ndarray:
    """
    Threshold gate: w = 1 if D(x) < τ, else 0
    
    Args:
        divergence: Divergence statistic D(x) ∈ [0, 1]
        stakes: Not used (kept for API consistency)
        tau: Threshold for trusting prior
    
    Returns:
        w ∈ {0, 1} binary trust indicator
    """
    divergence = np.asarray(divergence)
    
    if np.any(~np.isfinite(divergence)):
        raise ValueError("Divergence must be finite values")
    
    # Binary decision
    w = (divergence < tau).astype(float)
    
    return w


# =============================================================================
# METRICS
# =============================================================================

def calculate_ece(probs: np.ndarray, 
                  labels: np.ndarray, 
                  n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    
    Measures how well predicted probabilities match actual frequencies.
    
    Args:
        probs: Predicted probabilities (confidence in "accept")
        labels: Ground truth (1 if H is true, 0 otherwise)
        n_bins: Number of bins for calibration curve
    
    Returns:
        ECE (lower is better, 0 is perfect)
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    
    # Validate inputs
    if probs.shape != labels.shape:
        raise ValueError("Probs and labels must have same shape")
    
    if np.any(~np.isfinite(probs)) or np.any(~np.isfinite(labels)):
        raise ValueError("Probs and labels must be finite values")
    
    if len(probs) == 0:
        return 0.0
    
    # Bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        if i == n_bins - 1:
            # Last bin: include right boundary (fix for values exactly at 1.0)
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        else:
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        
        n_bin = np.sum(mask)
        
        if n_bin > 0:
            avg_confidence = np.mean(probs[mask])
            avg_accuracy = np.mean(labels[mask])
            ece += (n_bin / len(probs)) * abs(avg_confidence - avg_accuracy)
    
    return float(ece)


def calculate_auc_roc(labels: np.ndarray, 
                      scores: np.ndarray) -> float:
    """
    Area Under ROC Curve (AUC-ROC).
    
    Measures ranking quality of scores.
    
    Args:
        labels: Ground truth (1 = H true, 0 = H false)
        scores: Decision scores (higher = more likely to accept)
    
    Returns:
        AUC-ROC ∈ [0, 1] (0.5 = random, 1.0 = perfect)
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    # Validate inputs
    if labels.shape != scores.shape:
        raise ValueError("Labels and scores must have same shape")
    
    if len(labels) < 2:
        return 0.5  # Undefined for single sample
    
    # Check both classes exist
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.5  # No negative/positive class to measure
    
    try:
        auc = roc_auc_score(labels, scores)
        return float(auc)
    except Exception:
        return 0.5  # Fallback for edge cases


def calculate_brier_score(probs: np.ndarray, 
                          labels: np.ndarray) -> float:
    """
    Brier Score.
    
    Mean squared error between probabilities and labels.
    
    Args:
        probs: Predicted probabilities (confidence in "accept")
        labels: Ground truth (1 = H true, 0 = H false)
    
    Returns:
        Brier Score (lower is better, 0 is perfect)
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    
    # Validate inputs
    if probs.shape != labels.shape:
        raise ValueError("Probs and labels must have same shape")
    
    if np.any(~np.isfinite(probs)) or np.any(~np.isfinite(labels)):
        raise ValueError("Probs and labels must be finite values")
    
    if len(probs) == 0:
        return 0.0
    
    brier = np.mean((probs - labels) ** 2)
    return float(brier)


# =============================================================================
# DECISION SCORE CONVERSION
# =============================================================================

def convert_decision_to_probability(decision: str, 
                                    score: float,
                                    upper: float = 2.944,
                                    lower: float = -2.944) -> float:
    """
    Convert SPRT decision and score to probability-like confidence.
    
    Args:
        decision: "accept", "reject", or "budget_exhausted"
        score: Final log-likelihood ratio
        upper: Accept threshold
        lower: Reject threshold
    
    Returns:
        Probability-like value in [0, 1]
    """
    if decision == "accept":
        # Score exceeded upper bound — high confidence
        # Map score to [0.9, 1.0] based on how much it exceeded
        excess = (score - upper) / (abs(lower) + upper)
        prob = 0.9 + 0.1 * min(excess, 1.0)
        return float(np.clip(prob, 0.9, 1.0))
    
    elif decision == "reject":
        # Score went below lower bound — low confidence
        excess = (lower - score) / (abs(lower) + upper)
        prob = 0.1 - 0.1 * min(excess, 1.0)
        return float(np.clip(prob, 0.0, 0.1))
    
    else:  # budget_exhausted
        # Normalized score to [0, 1]
        total_range = upper - lower
        normalized = (score - lower) / total_range
        return float(np.clip(normalized, 0.0, 1.0))


# =============================================================================
# CALIBRATION PLOTTING HELPERS (FOR VISUALIZATION)
# =============================================================================

def get_calibration_curve(probs: np.ndarray, 
                          labels: np.ndarray, 
                          n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve points for plotting.
    
    Args:
        probs: Predicted probabilities
        labels: Ground truth
        n_bins: Number of bins
    
    Returns:
        Tuple of (bin_centers, avg_confidence, avg_accuracy)
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    avg_confs = []
    avg_accs = []
    
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        else:
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        
        n_bin = np.sum(mask)
        
        if n_bin > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            avg_confs.append(np.mean(probs[mask]))
            avg_accs.append(np.mean(labels[mask]))
        else:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            avg_confs.append(None)
            avg_accs.append(None)
    
    return (np.array(bin_centers), 
            np.array(avg_confs, dtype=float), 
            np.array(avg_accs, dtype=float))


# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

def validate_parameters(params: Dict[str, float], 
                        gate_type: str) -> None:
    """
    Validate gating function parameters before fitting.
    
    Args:
        params: Dictionary of parameter name → value
        gate_type: "logistic", "linear", or "threshold"
    
    Raises:
        ValueError: If parameters are invalid
    """
    if gate_type == "logistic":
        required = {"a", "b", "c"}
        for param_name in required:
            if param_name not in params:
                raise ValueError(f"Logistic gate requires parameter '{param_name}'")
            if not np.isfinite(params[param_name]):
                raise ValueError(f"Parameter '{param_name}' must be finite")
        
        # Warn if a is negative (opposite of intended behavior)
        if params.get("a", 1.0) < 0:
            print(f"Warning: a = {params['a']} is negative. "
                  f"Expected a > 0 for 'high divergence = low trust'")
    
    elif gate_type == "linear":
        required = {"alpha_coef", "beta_coef"}
        for param_name in required:
            if param_name not in params:
                raise ValueError(f"Linear gate requires parameter '{param_name}'")
            if not np.isfinite(params[param_name]):
                raise ValueError(f"Parameter '{param_name}' must be finite")
    
    elif gate_type == "threshold":
        required = {"tau"}
        for param_name in required:
            if param_name not in params:
                raise ValueError(f"Threshold gate requires parameter '{param_name}'")
            if not np.isfinite(params[param_name]):
                raise ValueError(f"Parameter '{param_name}' must be finite")
            if not (0.0 <= params[param_name] <= 1.0):
                raise ValueError(f"Threshold tau must be in [0, 1], got {params[param_name]}")
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")


# =============================================================================
# GATE FACTORY (SELECT BY NAME)
# =============================================================================

def get_gate_function(gate_type: str):
    """
    Factory function to get gate by name.
    
    Args:
        gate_type: "logistic", "linear", or "threshold"
    
    Returns:
        Gate function
    
    Raises:
        ValueError: If gate_type is unknown
    """
    gate_map = {
        "logistic": gate_logistic,
        "linear": gate_linear,
        "threshold": gate_threshold,
    }
    
    if gate_type not in gate_map:
        raise ValueError(f"Unknown gate type: {gate_type}. "
                        f"Supported: {list(gate_map.keys())}")
    
    return gate_map[gate_type]


# =============================================================================
# __all__ EXPORT LIST
# =============================================================================

__all__ = [
    # Gate functions
    "gate_logistic",
    "gate_linear",
    "gate_threshold",
    
    # Metrics
    "calculate_ece",
    "calculate_auc_roc",
    "calculate_brier_score",
    
    # Utilities
    "convert_decision_to_probability",
    "get_calibration_curve",
    "validate_parameters",
    "get_gate_function",
]
