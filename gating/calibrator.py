"""
Calibrator Module for TGAI Benchmark

Fits optimal gating parameters on Domain A data and evaluates transfer performance.

Capabilities:
- Fit logistic/linear/threshold gate parameters
- Cross-validation for robust hyperparameter tuning
- Multiple loss functions (ECE, accuracy, Brier score, weighted combo)
- Save/load fitted parameters
- Compare baseline vs calibrated performance
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from pathlib import Path
import json
import pickle

from .g_function import (
    gate_logistic, gate_linear, gate_threshold,
    calculate_ece, calculate_auc_roc, calculate_brier_score,
    convert_decision_to_probability, get_gate_function,
    validate_parameters
)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_domain_data(csv_path: str) -> pd.DataFrame:
    """
    Load Domain A data from CSV.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with trial results
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns missing
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    required_cols = [
        'w_gate', 'correct_blind', 'correct_informed', 'correct_blended',
        'divergence', 'stakes', 'score_blended'
    ]
    
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def loss_ece(params: np.ndarray, 
             divergence: np.ndarray, 
             stakes: np.ndarray,
             probs: np.ndarray,
             labels: np.ndarray,
             gate_type: str = "logistic") -> float:
    """
    Loss function: Expected Calibration Error (to minimize).
    
    Args:
        params: Gate parameters [a, b, c] for logistic, [alpha_coef, beta_coef] for linear
        divergence: Divergence values
        stakes: Stakes values
        probs: Base probability scores
        labels: Ground truth labels
        gate_type: "logistic", "linear", or "threshold"
        
    Returns:
        ECE value (lower is better)
    """
    gate_fn = get_gate_function(gate_type)
    
    try:
        w = gate_fn(divergence, stakes, *params)
        w = np.clip(w, 0.0, 1.0)
        
        # Weighted probabilities using gate weights
        calibrated_probs = w * probs + (1 - w) * 0.5
        
        ece = calculate_ece(calibrated_probs, labels)
        return ece
    except Exception:
        return 1e6  # Large penalty for invalid params

def loss_accuracy(params: np.ndarray,
                  divergence: np.ndarray,
                  stakes: np.ndarray,
                  scores: np.ndarray,
                  labels: np.ndarray,
                  gate_type: str = "logistic") -> float:
    """
    Loss function: Negative accuracy (to minimize = maximize accuracy).
    
    Args:
        params: Gate parameters
        divergence: Divergence values
        stakes: Stakes values
        scores: Decision scores
        labels: Ground truth labels
        gate_type: Gate type
        
    Returns:
        Negative accuracy (so minimization maximizes accuracy)
    """
    gate_fn = get_gate_function(gate_type)
    
    try:
        w = gate_fn(divergence, stakes, *params)
        w = np.clip(w, 0.0, 1.0)
        
        # Calibrated scores
        calibrated_scores = w * scores + (1 - w) * np.mean(scores)
        
        # Threshold at 0 for binary decisions
        predictions = (calibrated_scores >= 0).astype(int)
        
        accuracy = np.mean(predictions == labels)
        return -accuracy  # Negate for minimization
    except Exception:
        return 1e6

def loss_brier(params: np.ndarray,
               divergence: np.ndarray,
               stakes: np.ndarray,
               probs: np.ndarray,
               labels: np.ndarray,
               gate_type: str = "logistic") -> float:
    """
    Loss function: Brier Score (to minimize).
    
    Args:
        params: Gate parameters
        divergence: Divergence values
        stakes: Stakes values
        probs: Base probability scores
        labels: Ground truth labels
        gate_type: Gate type
        
    Returns:
        Brier Score (lower is better)
    """
    gate_fn = get_gate_function(gate_type)
    
    try:
        w = gate_fn(divergence, stakes, *params)
        w = np.clip(w, 0.0, 1.0)
        
        calibrated_probs = w * probs + (1 - w) * 0.5
        
        brier = calculate_brier_score(calibrated_probs, labels)
        return brier
    except Exception:
        return 1e6

def loss_combined(params: np.ndarray,
                  divergence: np.ndarray,
                  stakes: np.ndarray,
                  probs: np.ndarray,
                  scores: np.ndarray,
                  labels: np.ndarray,
                  gate_type: str = "logistic",
                  ece_weight: float = 0.5,
                  accuracy_weight: float = 0.3,
                  brier_weight: float = 0.2) -> float:
    """
    Combined loss: weighted sum of ECE, negative accuracy, and Brier score.
    
    Args:
        params: Gate parameters
        divergence: Divergence values
        stakes: Stakes values
        probs: Base probability scores
        scores: Decision scores
        labels: Ground truth labels
        gate_type: Gate type
        ece_weight: Weight for ECE component
        accuracy_weight: Weight for accuracy component
        brier_weight: Weight for Brier score component
        
    Returns:
        Combined loss value
    """
    ece_loss = loss_ece(params, divergence, stakes, probs, labels, gate_type)
    acc_loss = loss_accuracy(params, divergence, stakes, scores, labels, gate_type)
    brier_loss = loss_brier(params, divergence, stakes, probs, labels, gate_type)
    
    # Normalize losses to similar scale
    ece_loss = ece_loss / 0.1  # ECE typically < 0.1
    acc_loss = acc_loss + 0.5  # Accuracy typically > 0.5
    brier_loss = brier_loss / 0.25  # Brier typically < 0.25
    
    combined = (ece_weight * ece_loss + 
                accuracy_weight * acc_loss + 
                brier_weight * brier_loss)
    
    return combined

# =============================================================================
# CALIBRATION ENGINE
# =============================================================================

@dataclass
class CalibrationResult:
    """Container for calibration results."""
    
    gate_type: str
    params: Dict[str, float]
    baseline_ece: float
    calibrated_ece: float
    baseline_accuracy: float
    calibrated_accuracy: float
    baseline_auc: float
    calibrated_auc: float
    improvement_ece: float
    improvement_accuracy: float
    fit_method: str
    n_samples: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "gate_type": self.gate_type,
            "params": self.params,
            "baseline_ece": self.baseline_ece,
            "calibrated_ece": self.calibrated_ece,
            "improvement_ece_pct": ((self.baseline_ece - self.calibrated_ece) / 
                                     self.baseline_ece * 100) if self.baseline_ece > 0 else 0,
            "baseline_accuracy": self.baseline_accuracy,
            "calibrated_accuracy": self.calibrated_accuracy,
            "improvement_accuracy_pct": ((self.calibrated_accuracy - 
                                          self.baseline_accuracy) / 
                                         self.baseline_accuracy * 100) if self.baseline_accuracy > 0 else 0,
            "baseline_auc": self.baseline_auc,
            "calibrated_auc": self.calibrated_auc,
            "fit_method": self.fit_method,
            "n_samples": self.n_samples
        }

def fit_params_simple(params: np.ndarray,
                      loss_fn: callable,
                      divergence: np.ndarray,
                      stakes: np.ndarray,
                      probs: np.ndarray,
                      scores: np.ndarray,
                      labels: np.ndarray,
                      gate_type: str,
                      bounds: Optional[List[Tuple[float, float]]] = None,
                      method: str = "L-BFGS-B") -> Tuple[np.ndarray, float]:
    """
    Simple gradient-based optimization.
    
    Args:
        params: Initial parameter guess
        loss_fn: Loss function
        divergence: Divergence values
        stakes: Stakes values
        probs: Probability scores
        scores: Decision scores
        labels: Ground truth
        gate_type: Gate type
        bounds: Parameter bounds
        method: Optimization method
        
    Returns:
        Tuple of (optimized_params, final_loss)
    """
    result = minimize(
        fun=lambda p: loss_fn(p, divergence, stakes, probs, scores, labels, gate_type),
        x0=params,
        method=method,
        bounds=bounds
    )
    
    return result.x, result.fun

def fit_params_global(loss_fn: callable,
                      divergence: np.ndarray,
                      stakes: np.ndarray,
                      probs: np.ndarray,
                      scores: np.ndarray,
                      labels: np.ndarray,
                      gate_type: str,
                      n_population: int = 50,
                      max_iter: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Global optimization using differential evolution.
    More robust but slower than gradient-based methods.
    
    Args:
        loss_fn: Loss function
        divergence: Divergence values
        stakes: Stakes values
        probs: Probability scores
        scores: Decision scores
        labels: Ground truth
        gate_type: Gate type
        n_population: Population size for DE
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (optimized_params, final_loss)
    """
    n_params = 3 if gate_type == "logistic" else (1 if gate_type == "threshold" else 2)
    
    bounds = [(-5.0, 5.0)] * n_params  # Reasonable range for gate parameters
    
    result = differential_evolution(
        func=lambda p: loss_fn(p, divergence, stakes, probs, scores, labels, gate_type),
        bounds=bounds,
        popsize=n_population,
        maxiter=max_iter,
        tol=1e-6,
        polish=True
    )
    
    return result.x, result.fun

# =============================================================================
# CROSS-VALIDATION
# =============================================================================

@dataclass
class CrossValidationResult:
    """Container for cross-validation results."""
    
    fold_metrics: List[Dict]
    mean_ece: float
    std_ece: float
    mean_accuracy: float
    std_accuracy: float
    mean_auc: float
    std_auc: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "fold_metrics": self.fold_metrics,
            "mean_ece": self.mean_ece,
            "std_ece": self.std_ece,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "mean_auc": self.mean_auc,
            "std_auc": self.std_auc
        }

def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] via sigmoid, but only if they aren't
    already probabilities. Kept as a shared helper so cross_validate
    and calibrate_gates apply the exact same rule.
    """
    if probs.min() < 0 or probs.max() > 1:
        return 1.0 / (1.0 + np.exp(-probs))
    return probs

def cross_validate(data_df: pd.DataFrame,
                   n_folds: int = 5,
                   gate_type: str = "logistic",
                   loss_fn_name: str = "combined",
                   random_seed: int = 42,
                   n_samples: int = 500) -> CrossValidationResult:
    """
    Perform k-fold cross-validation for gate fitting.
    
    Args:
        data_df: DataFrame with trial results
        n_folds: Number of CV folds
        gate_type: "logistic", "linear", or "threshold"
        loss_fn_name: "ece", "accuracy", "brier", or "combined"
        random_seed: Random seed for reproducibility
        n_samples: Samples per fold for fitting (subsampling for speed)
        
    Returns:
        CrossValidationResult with fold-level and aggregate metrics
    """
    rng = np.random.RandomState(random_seed)
    indices = rng.permutation(len(data_df))
    
    fold_size = len(indices) // n_folds
    fold_metrics = []
    
    loss_fn_map = {
        "ece": loss_ece,
        "accuracy": loss_accuracy,
        "brier": loss_brier,
        "combined": loss_combined
    }
    loss_fn = loss_fn_map.get(loss_fn_name, loss_combined)
    
    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else len(indices)
        
        val_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        train_df = data_df.iloc[train_indices]
        val_df = data_df.iloc[val_indices]
        
        # Subsample for faster fitting
        if len(train_df) > n_samples:
            train_df = train_df.sample(n=n_samples, random_state=random_seed + fold)
        
        # Prepare features
        divergence = train_df['divergence'].values
        stakes = train_df['stakes'].values
        probs = train_df['score_blended'].values
        scores = train_df['score_blended'].values
        labels = train_df['correct_blended'].values
        
        # Convert probs to [0, 1] range if needed
        probs = _normalize_probs(probs)
        
        # Fit parameters
        n_params = 3 if gate_type == "logistic" else (1 if gate_type == "threshold" else 2)
        initial_params = rng.randn(n_params) * 0.5
        
        try:
            # Try global optimization first
            opt_params, _ = fit_params_global(
                loss_fn, divergence, stakes, probs, scores, labels, 
                gate_type, n_population=20, max_iter=500
            )
        except Exception:
            # Fall back to simple optimization
            opt_params, _ = fit_params_simple(
                initial_params, loss_fn, divergence, stakes, 
                probs, scores, labels, gate_type
            )
        
        # Evaluate on validation set
        val_div = val_df['divergence'].values
        val_stakes = val_df['stakes'].values
        val_probs = val_df['score_blended'].values
        val_labels = val_df['correct_blended'].values
        
        val_probs = _normalize_probs(val_probs)
        
        gate_fn = get_gate_function(gate_type)
        w = gate_fn(val_div, val_stakes, *opt_params)
        w = np.clip(w, 0.0, 1.0)
        
        calibrated_probs = w * val_probs + (1 - w) * 0.5
        
        fold_results = {
            "fold": fold + 1,
            "params": opt_params.tolist(),
            "ece": calculate_ece(calibrated_probs, val_labels),
            "accuracy": np.mean((calibrated_probs >= 0.5).astype(int) == val_labels),
            "auc": calculate_auc_roc(val_labels, calibrated_probs)
        }
        
        fold_metrics.append(fold_results)
    
    # Compute aggregate statistics
    eces = [m["ece"] for m in fold_metrics]
    accs = [m["accuracy"] for m in fold_metrics]
    aucs = [m["auc"] for m in fold_metrics]
    
    cv_result = CrossValidationResult(
        fold_metrics=fold_metrics,
        mean_ece=np.mean(eces),
        std_ece=np.std(eces),
        mean_accuracy=np.mean(accs),
        std_accuracy=np.std(accs),
        mean_auc=np.mean(aucs),
        std_auc=np.std(aucs)
    )
    
    return cv_result

# =============================================================================
# MAIN CALIBRATION FUNCTION
# =============================================================================

def calibrate_gates(data_csv: str,
                    gate_types: List[str] = ["logistic"],
                    loss_fn_name: str = "combined",
                    n_cv_folds: int = 5,
                    random_seed: int = 42,
                    output_dir: str = "results/calibration") -> Dict[str, CalibrationResult]:
    """
    Main calibration function: fit all gate types and compare.
    
    Args:
        data_csv: Path to Domain A CSV
        gate_types: List of gate types to evaluate
        loss_fn_name: Which loss function to optimize
        n_cv_folds: Number of cross-validation folds
        random_seed: Random seed
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping gate_type → CalibrationResult
    """
    # Load data
    data_df = load_domain_data(data_csv)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    calibration_results = {}
    
    for gate_type in gate_types:
        print(f"Fitting {gate_type} gate...")
        
        # Run cross-validation
        cv_result = cross_validate(
            data_df, n_folds=n_cv_folds, gate_type=gate_type,
            loss_fn_name=loss_fn_name, random_seed=random_seed
        )
        
        # Get best parameters (from median fold)
        fold_metrics = sorted(cv_result.fold_metrics, key=lambda x: x["ece"])
        best_fold = fold_metrics[len(fold_metrics) // 2]
        best_params = best_fold["params"]
        
        # Prepare parameter dict
        if gate_type == "logistic":
            params_dict = {"a": best_params[0], "b": best_params[1], "c": best_params[2]}
        elif gate_type == "linear":
            params_dict = {"alpha_coef": best_params[0], "beta_coef": best_params[1]}
        else:  # threshold
            params_dict = {"tau": best_params[0]}
        
        # Compute baseline metrics (using original uncalibrated data).
        # Use the same normalization rule as cross_validate so baseline
        # and calibrated numbers are on the same scale.
        baseline_probs = _normalize_probs(data_df['score_blended'].values.astype(float))
        baseline_labels = data_df['correct_blended'].values
        
        baseline_ece = calculate_ece(baseline_probs, baseline_labels)
        baseline_accuracy = np.mean((baseline_probs >= 0.5).astype(int) == baseline_labels)
        baseline_auc = calculate_auc_roc(baseline_labels, baseline_probs)
        
        # Create calibration result
        cal_result = CalibrationResult(
            gate_type=gate_type,
            params=params_dict,
            baseline_ece=baseline_ece,
            calibrated_ece=cv_result.mean_ece,
            baseline_accuracy=baseline_accuracy,
            calibrated_accuracy=cv_result.mean_accuracy,
            baseline_auc=baseline_auc,
            calibrated_auc=cv_result.mean_auc,
            improvement_ece=baseline_ece - cv_result.mean_ece,
            improvement_accuracy=cv_result.mean_accuracy - baseline_accuracy,
            fit_method="differential_evolution",
            n_samples=len(data_df)
        )
        
        calibration_results[gate_type] = cal_result
        
        # Save results
        result_path = Path(output_dir) / f"{gate_type}_results.json"
        with open(result_path, 'w') as f:
            json.dump({
                **cal_result.to_dict(),
                "cross_validation": cv_result.to_dict()
            }, f, indent=2)
        
        # Print summary
        print(f"  {gate_type}: ECE {baseline_ece:.4f} → {cv_result.mean_ece:.4f} "
              f"(Δ={cal_result.improvement_ece:.4f})")
        print(f"    Accuracy: {baseline_accuracy:.4f} → {cv_result.mean_accuracy:.4f}")
        print(f"    Params: {params_dict}")
    
    # Save summary
    summary_path = Path(output_dir) / "calibration_summary.json"
    summary = {
        gate_type: result.to_dict()
        for gate_type, result in calibration_results.items()
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nCalibration complete. Results saved to: {output_dir}/")
    
    return calibration_results

# =============================================================================
# LOAD CALIBRATED PARAMETERS
# =============================================================================

def load_calibrated_params(gate_type: str, 
                           results_dir: str = "results/calibration") -> Dict[str, float]:
    """
    Load previously fitted gate parameters.
    
    Args:
        gate_type: "logistic", "linear", or "threshold"
        results_dir: Directory with calibration results
        
    Returns:
        Dictionary of parameter names → values
        
    Raises:
        FileNotFoundError: If results file doesn't exist
    """
    result_path = Path(results_dir) / f"{gate_type}_results.json"
    
    if not result_path.exists():
        raise FileNotFoundError(f"Calibration results not found: {result_path}")
    
    with open(result_path, 'r') as f:
        data = json.load(f)
    
    return data["params"]

def save_calibration_pickle(results: Dict[str, CalibrationResult],
                            filepath: str = "results/calibration.pkl"):
    """Save calibration results to pickle file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

def load_calibration_pickle(filepath: str = "results/calibration.pkl") -> Dict[str, CalibrationResult]:
    """Load calibration results from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# =============================================================================
# __all__ EXPORT LIST
# =============================================================================

__all__ = [
    # Data loading
    "load_domain_data",
    
    # Loss functions
    "loss_ece", "loss_accuracy", "loss_brier", "loss_combined",
    
    # Calibration engine
    "fit_params_simple", "fit_params_global",
    "CalibrationResult",
    
    # Cross-validation
    "CrossValidationResult", "cross_validate",
    
    # Main calibration
    "calibrate_gates",
    
    # Load/save
    "load_calibrated_params", "save_calibration_pickle", "load_calibration_pickle"
]
