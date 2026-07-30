"""
Domain A: Synthetic Sequential Testing (TGAI Benchmark)

Implements dual-track SPRT with trust-gated auxiliary information.

Architecture:
    Evidence Stream → Track 1: Prior-Blind SPRT    → score_blind
                   → Track 2: Prior-Informed SPRT  → score_informed
                                                          ↓
                   w = g(divergence, stakes)
                                                          ↓
                   Final = w · score_informed + (1-w) · score_blind
                                                          ↓
                   Decision: accept / reject / budget_exhausted

Each trial randomly varies:
    - Ground truth (H true or false)
    - Prior quality (aligned, stale, adversarial)
    - Effect strength (weak, moderate, strong)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass
import argparse
import os

@dataclass
class TrialConfig:
    """Per-trial configuration (randomly sampled for each trial)."""
    
    coin_bias_true: float = 0.0
    coin_bias_prior: float = 0.0
    effect_strength: float = 0.0
    alpha: float = 0.05
    beta: float = 0.05
    max_steps: int = 200
    random_seed: int = 42

@dataclass
class BenchmarkConfig:
    """Global benchmark configuration."""
    
    n_trials: int = 1000
    tosses_per_trial: int = 200
    random_seed: int = 42
    
    bias_range: Tuple[float, float] = (0.3, 0.8)
    effect_range: Tuple[float, float] = (0.05, 0.45)
    prior_noise_std: float = 0.15
    prior_adversarial_prob: float = 0.2

def sample_trial_config(bench_config: BenchmarkConfig,
                        rng: np.random.Generator,
                        trial_idx: int) -> TrialConfig:
    """Randomly sample per-trial configuration."""
    coin_bias_true = rng.uniform(*bench_config.bias_range)
    effect_strength = rng.uniform(*bench_config.effect_range)
    
    if rng.random() < bench_config.prior_adversarial_prob:
        coin_bias_prior = 1.0 - coin_bias_true + rng.normal(0, bench_config.prior_noise_std)
    else:
        coin_bias_prior = coin_bias_true + rng.normal(0, bench_config.prior_noise_std)
    
    coin_bias_prior = np.clip(coin_bias_prior, 0.01, 0.99)
    
    # FIX: alpha=0.05, NOT bench_config.random_seed
    return TrialConfig(
        coin_bias_true=coin_bias_true,
        coin_bias_prior=coin_bias_prior,
        effect_strength=effect_strength,
        alpha=0.05,              # FIXED: Was bench_config.random_seed (42)
        beta=0.05,
        max_steps=bench_config.tosses_per_trial,
        random_seed=bench_config.random_seed + trial_idx,
    )

# =============================================================================
# CORE SPRT ENGINE
# =============================================================================

def calculate_log_lr(toss: int, p_h: float, p_not_h: float) -> float:
    """Single-observation log-likelihood ratio."""
    if toss == 1:
        lik_h, lik_not_h = p_h, p_not_h
    else:
        lik_h, lik_not_h = 1 - p_h, 1 - p_not_h
    
    lik_h = max(lik_h, 1e-10)
    lik_not_h = max(lik_not_h, 1e-10)
    return np.log(lik_h / lik_not_h)

def run_track_sprt(tosses: np.ndarray,
                   p_h: float,
                   p_not_h: float,
                   upper: float,
                   lower: float,
                   max_steps: int,
                   prior_weight: float = 0.0,
                   prior_bias: float = 0.5) -> Tuple[str, int, float]:
    """SPRT with optional prior influence."""
    p_h_effective = (1 - prior_weight) * p_h + prior_weight * prior_bias
    p_not_h_effective = (1 - prior_weight) * p_not_h + prior_weight * (1 - prior_bias)
    
    log_lr = 0.0
    for i, toss in enumerate(tosses[:max_steps]):
        log_lr += calculate_log_lr(toss, p_h_effective, p_not_h_effective)
        
        if log_lr >= upper:
            return "accept", i + 1, log_lr
        elif log_lr <= lower:
            return "reject", i + 1, log_lr
    
    decision = "accept" if log_lr >= 0 else "reject"
    return decision, max_steps, log_lr

# =============================================================================
# GATING FUNCTION
# =============================================================================

def gate_logistic(divergence: float, stakes: float,
                  a: float = 1.0, b: float = 1.0, c: float = 0.0) -> float:
    """g(x) = σ(a·D(x) - b·S(x) - c)"""
    z = a * divergence - b * stakes - c
    return 1.0 / (1.0 + np.exp(-z))

# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def compute_divergence(prior: float, mle: float) -> float:
    """D(x) = |prior - MLE| normalized to [0, 1]."""
    return min(abs(prior - mle), 1.0)

def compute_stakes(alpha: float, beta: float) -> float:
    """S(x) = |log(α/β)| — asymmetry between Type I and Type II error costs."""
    return abs(np.log(alpha / beta))

# =============================================================================
# SINGLE TRIAL EXECUTION
# =============================================================================

def run_single_trial(config: TrialConfig, bench_config: BenchmarkConfig) -> Dict:
    """Run one complete trial with dual-track SPRT + trust gating."""
    rng = np.random.default_rng(seed=config.random_seed)
    
    h_is_true = config.coin_bias_true > 0.5
    p_h = 0.5 + config.effect_strength / 2
    p_not_h = 0.5 - config.effect_strength / 2
    
    tosses = rng.binomial(n=1, p=config.coin_bias_true,
                          size=bench_config.tosses_per_trial).astype(int)
    mle = np.mean(tosses[:config.max_steps])
    
    divergence = compute_divergence(config.coin_bias_prior, mle)
    stakes = compute_stakes(config.alpha, config.beta)
    w = gate_logistic(divergence, stakes)
    
    upper = np.log((1 - config.beta) / config.alpha)
    lower = np.log(config.beta / (1 - config.alpha))
    
    decision_blind, steps_blind, score_blind = run_track_sprt(
        tosses, p_h, p_not_h, upper, lower, config.max_steps,
        prior_weight=0.0, prior_bias=config.coin_bias_prior
    )
    
    decision_informed, steps_informed, score_informed = run_track_sprt(
        tosses, p_h, p_not_h, upper, lower, config.max_steps,
        prior_weight=1.0, prior_bias=config.coin_bias_prior
    )
    
    blended_score = w * score_informed + (1 - w) * score_blind
    blended_decision = "accept" if blended_score >= 0 else "reject"
    blended_steps = int(w * steps_informed + (1 - w) * steps_blind)
    
    correct_blind = (decision_blind == "accept") == h_is_true
    correct_informed = (decision_informed == "accept") == h_is_true
    correct_blended = (blended_decision == "accept") == h_is_true
    
    return {
        "divergence": divergence,
        "stakes": stakes,
        "w_gate": w,
        "h_is_true": int(h_is_true),
        "coin_bias_true": config.coin_bias_true,
        "coin_bias_prior": config.coin_bias_prior,
        "effect_strength": config.effect_strength,
        "mle_estimate": mle,
        "decision_blind": decision_blind,
        "steps_blind": steps_blind,
        "score_blind": score_blind,
        "correct_blind": int(correct_blind),
        "decision_informed": decision_informed,
        "steps_informed": steps_informed,
        "score_informed": score_informed,
        "correct_informed": int(correct_informed),
        "decision_blended": blended_decision,
        "steps_blended": blended_steps,
        "score_blended": blended_score,
        "correct_blended": int(correct_blended),
    }

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(bench_config: BenchmarkConfig = None) -> pd.DataFrame:
    """Run full benchmark with randomized per-trial configs."""
    if bench_config is None:
        bench_config = BenchmarkConfig()
    
    master_rng = np.random.default_rng(seed=bench_config.random_seed)
    
    print(f"Running {bench_config.n_trials} sequential testing trials...")
    print(f"  Bias range: {bench_config.bias_range}")
    print(f"  Effect range: {bench_config.effect_range}")
    print(f"  Prior adversarial prob: {bench_config.prior_adversarial_prob}")
    print()
    
    results = []
    for i in range(bench_config.n_trials):
        trial_config = sample_trial_config(bench_config, master_rng, i)
        result = run_single_trial(trial_config, bench_config)
        result["trial_id"] = i
        results.append(result)
        
        if (i + 1) % 100 == 0:
            acc = np.mean([r["correct_blended"] for r in results])
            print(f"  Trial {i+1}/{bench_config.n_trials} | Blended acc so far: {acc:.3f}")
    
    df = pd.DataFrame(results)
    
    print(f"\nBenchmark complete. {len(df)} trials generated.")
    print(f"\nClass balance:")
    print(f"  H=True: {df['h_is_true'].sum()} ({df['h_is_true'].mean():.1%})")
    print(f"  H=False: {(1-df['h_is_true']).sum()} ({(1-df['h_is_true']).mean():.1%})")
    print(f"\nAccuracy comparison:")
    print(f"  Prior-blind:   {df['correct_blind'].mean():.3f}")
    print(f"  Prior-informed: {df['correct_informed'].mean():.3f}")
    print(f"  Blended (gate): {df['correct_blended'].mean():.3f}")
    print(f"\nFeature ranges:")
    print(f"  Divergence: [{df['divergence'].min():.3f}, {df['divergence'].max():.3f}]")
    print(f"  Stakes:     [{df['stakes'].min():.3f}, {df['stakes'].max():.3f}]")
    print(f"  w (gate):   [{df['w_gate'].min():.3f}, {df['w_gate'].max():.3f}]")
    
    return df

def split_data(df: pd.DataFrame,
               train_frac: float = 0.7,
               val_frac: float = 0.15,
               random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test with seeded reproducibility."""
    rng = np.random.RandomState(random_seed)
    indices = rng.permutation(len(df))
    
    n_train = int(train_frac * len(df))
    n_val = int(val_frac * len(df))
    
    return (df.iloc[indices[:n_train]].reset_index(drop=True),
            df.iloc[indices[n_train:n_train+n_val]].reset_index(drop=True),
            df.iloc[indices[n_train+n_val:]].reset_index(drop=True))

def calculate_ece(predictions: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
        n_bin = np.sum(mask)
        if n_bin > 0:
            ece += (n_bin / len(predictions)) * abs(
                np.mean(predictions[mask]) - np.mean(actuals[mask])
            )
    return ece

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain A: Sequential Testing Benchmark (TGAI)")
    parser.add_argument("--n-trials", type=int, default=1000, help="Number of trials")
    parser.add_argument("--tosses-per-trial", type=int, default=200, help="Tosses per trial")
    parser.add_argument("--bias-min", type=float, default=0.3, help="Min true coin bias")
    parser.add_argument("--bias-max", type=float, default=0.8, help="Max true coin bias")
    parser.add_argument("--effect-min", type=float, default=0.05, help="Min effect strength")
    parser.add_argument("--effect-max", type=float, default=0.45, help="Max effect strength")
    parser.add_argument("--adversarial-prob", type=float, default=0.2, help="Prob of adversarial prior")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--output", type=str, default="results/domain_a_data.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        n_trials=args.n_trials,
        tosses_per_trial=args.tosses_per_trial,
        random_seed=args.seed,
        bias_range=(args.bias_min, args.bias_max),
        effect_range=(args.effect_min, args.effect_max),
        prior_adversarial_prob=args.adversarial_prob,
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    df = run_benchmark(config)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
