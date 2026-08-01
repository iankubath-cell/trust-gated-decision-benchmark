"""
Domain B: Sensor Fusion (KITTI-style) Benchmark for TGAI

Simulates LiDAR + Camera obstacle detection with map-based prior information.

Scenario:
    A vehicle navigates with two sensors (LiDAR, Camera) and an HD map prior.
    Each "reading" accumulates evidence about whether an obstacle is ahead.

    Track 1 (Prior-Blind): Uses only sensor likelihood ratio (log_lr starts at 0)
    Track 2 (Prior-Informed): Injects map prior as initial log-odds,
        then accumulates the SAME sensor likelihoods
    Gate w = g(divergence, stakes) blends the two tracks.

Data Modes:
    - "synthetic": Generate KITTI-like sensor readings (default, no download needed)
    - "kitti": Load real KITTI odometry data (requires local KITTI dataset)

Prior Quality Variation (mirrors Domain A):
    - "aligned": Map agrees with ground truth (prior ~ truth)
    - "stale": Map is outdated (prior has noise)
    - "adversarial": Map is wrong (prior inverted)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import argparse
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SensorTrialConfig:
    """Per-trial configuration for sensor fusion."""
    obstacle_present: bool = True        # Ground truth
    map_prior_belief: float = 0.5        # Map's belief that obstacle exists
    lidar_reliability: float = 0.8       # P(correct reading | obstacle)
    camera_reliability: float = 0.7      # P(correct reading | obstacle)
    prior_quality: str = "aligned"       # "aligned", "stale", "adversarial"
    alpha: float = 0.05                  # SPRT Type I error
    beta: float = 0.05                   # SPRT Type II error
    max_steps: int = 50                  # Max sensor readings per trial
    random_seed: int = 42


@dataclass
class SensorBenchmarkConfig:
    """Global configuration for sensor fusion benchmark."""
    n_trials: int = 1000
    readings_per_trial: int = 50
    random_seed: int = 42

    obstacle_prob: float = 0.55          # P(obstacle exists in a trial)

    # Sensor reliability ranges (simulating varying conditions)
    lidar_reliability_range: Tuple[float, float] = (0.6, 0.95)
    camera_reliability_range: Tuple[float, float] = (0.5, 0.9)

    # Prior quality distribution
    aligned_prob: float = 0.5            # P(prior is good)
    stale_prob: float = 0.3              # P(prior is outdated)
    adversarial_prob: float = 0.2        # P(prior is wrong)

    # Noise parameters
    stale_noise_std: float = 0.15
    prior_smoothing: float = 0.1         # Smoothing for prior belief

    # SPRT error rates — sampled per trial from these ranges so that
    # `stakes = |log(alpha/beta)|` actually varies across trials.
    sprt_alpha_range: Tuple[float, float] = (0.01, 0.1)
    sprt_beta_range: Tuple[float, float] = (0.01, 0.1)

    # Data mode
    data_mode: str = "synthetic"         # "synthetic" or "kitti"
    kitti_path: Optional[str] = None     # Path to KITTI data if using real data


# =============================================================================
# SYNTHETIC SENSOR DATA GENERATION
# =============================================================================

def sample_sensor_trial_config(
    bench_config: SensorBenchmarkConfig,
    rng: np.random.Generator,
    trial_idx: int
) -> SensorTrialConfig:
    """Randomly sample per-trial configuration."""
    obstacle_present = rng.random() < bench_config.obstacle_prob

    lidar_rel = rng.uniform(*bench_config.lidar_reliability_range)
    camera_rel = rng.uniform(*bench_config.camera_reliability_range)

    # Determine prior quality
    rand_val = rng.random()
    if rand_val < bench_config.aligned_prob:
        prior_quality = "aligned"
    elif rand_val < bench_config.aligned_prob + bench_config.stale_prob:
        prior_quality = "stale"
    else:
        prior_quality = "adversarial"

    # Generate map prior belief based on quality
    true_belief = 1.0 if obstacle_present else 0.0

    if prior_quality == "aligned":
        map_prior_belief = true_belief + rng.normal(0, bench_config.prior_smoothing)
    elif prior_quality == "stale":
        map_prior_belief = true_belief + rng.normal(0, bench_config.stale_noise_std)
    else:  # adversarial
        map_prior_belief = (1.0 - true_belief) + rng.normal(0, bench_config.prior_smoothing)

    map_prior_belief = np.clip(map_prior_belief, 0.01, 0.99)

    # Sample per-trial SPRT error rates so stakes varies across trials
    alpha = rng.uniform(*bench_config.sprt_alpha_range)
    beta = rng.uniform(*bench_config.sprt_beta_range)

    return SensorTrialConfig(
        obstacle_present=obstacle_present,
        map_prior_belief=map_prior_belief,
        lidar_reliability=lidar_rel,
        camera_reliability=camera_rel,
        prior_quality=prior_quality,
        alpha=alpha,
        beta=beta,
        max_steps=bench_config.readings_per_trial,
        random_seed=bench_config.random_seed + trial_idx,
    )


def generate_sensor_readings(
    config: SensorTrialConfig,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic LiDAR and Camera readings.

    Each reading is binary: 1 = "obstacle detected", 0 = "no obstacle".

    P(detect | obstacle_present) = sensor_reliability
    P(detect | no obstacle) = 1 - sensor_reliability

    Returns:
        Tuple of (lidar_readings, camera_readings), each shape (max_steps,)
    """
    n = config.max_steps

    if config.obstacle_present:
        p_detect_lidar = config.lidar_reliability
        p_detect_camera = config.camera_reliability
    else:
        p_detect_lidar = 1.0 - config.lidar_reliability
        p_detect_camera = 1.0 - config.camera_reliability

    # Add some correlation between sensors (shared environmental conditions)
    env_noise = rng.normal(0, 0.05, size=n)
    p_detect_lidar = np.clip(p_detect_lidar + env_noise, 0.01, 0.99)
    p_detect_camera = np.clip(p_detect_camera + env_noise, 0.01, 0.99)

    lidar_readings = rng.binomial(n=1, p=p_detect_lidar, size=n).astype(int)
    camera_readings = rng.binomial(n=1, p=p_detect_camera, size=n).astype(int)

    return lidar_readings, camera_readings


# =============================================================================
# KITTI DATA LOADER (OPTIONAL)
# =============================================================================

def load_kitti_data(kitti_path: str, max_frames: int = 1000) -> Optional[pd.DataFrame]:
    """
    Attempt to load real KITTI odometry data.

    Expected structure:
        kitti_path/
        ├── poses/           # Ground truth poses
        ├── detections/      # Object detections
        └── sequences/       # Sensor data sequences

    Returns DataFrame or None if not found.
    """
    kitti_path = os.path.expanduser(kitti_path)

    if not os.path.isdir(kitti_path):
        print(f"  KITTI path not found: {kitti_path}")
        return None

    det_path = os.path.join(kitti_path, "detections")
    if not os.path.isdir(det_path):
        print(f"  KITTI detections folder not found: {det_path}")
        return None

    det_files = sorted([f for f in os.listdir(det_path) if f.endswith(".txt")])

    if len(det_files) == 0:
        print(f"  No detection files found in: {det_path}")
        return None

    print(f"  Loading {min(len(det_files), max_frames)} frames from KITTI...")

    records = []
    for i, fname in enumerate(det_files[:max_frames]):
        fpath = os.path.join(det_path, fname)
        try:
            with open(fpath, "r") as f:
                lines = f.readlines()

            obstacle_detected = 1 if len(lines) > 0 else 0
            n_objects = len(lines)

            # Very simple proxy confidences (bounded [0,1])
            lidar_conf = min(n_objects / 5.0, 1.0)
            camera_conf = min(n_objects / 6.0, 1.0) if n_objects > 0 else 0.5

            records.append({
                "frame_id": i,
                "obstacle_detected": obstacle_detected,
                "lidar_conf": lidar_conf,
                "camera_conf": camera_conf,
            })
        except Exception as e:
            print(f"  Warning: Could not parse {fname}: {e}")
            continue

    return pd.DataFrame(records) if records else None


# =============================================================================
# SPRT FOR SENSOR FUSION (FIXED PRIOR INJECTION)
# =============================================================================

def calculate_sensor_log_lr(
    lidar_reading: int,
    camera_reading: int,
    p_h_lidar: float,
    p_not_h_lidar: float,
    p_h_camera: float,
    p_not_h_camera: float
) -> float:
    """
    Combined log-likelihood ratio from LiDAR + Camera readings.

    Assumes conditional independence given H:
        LLR = log P(lidar|H) + log P(camera|H) - log P(lidar|¬H) - log P(camera|¬H)
    """
    def safe_llr(reading: int, p_h: float, p_not_h: float) -> float:
        p_h = max(p_h, 1e-10)
        p_not_h = max(p_not_h, 1e-10)
        if reading == 1:
            return np.log(p_h / p_not_h)
        return np.log((1 - p_h) / (1 - p_not_h))

    return (
        safe_llr(lidar_reading, p_h_lidar, p_not_h_lidar)
        + safe_llr(camera_reading, p_h_camera, p_not_h_camera)
    )


def run_sensor_sprt(
    lidar_readings: np.ndarray,
    camera_readings: np.ndarray,
    p_h_lidar: float,
    p_not_h_lidar: float,
    p_h_camera: float,
    p_not_h_camera: float,
    upper: float,
    lower: float,
    max_steps: int,
    prior_weight: float = 0.0,
    prior_belief: float = 0.5
) -> Tuple[str, int, float]:
    """
    SPRT on fused sensor data with prior injected as initial log-odds.

    The prior enters as an initial log-odds offset:
        log_lr_start = prior_weight * log(prior_belief / (1 - prior_belief))

    Then genuine sensor likelihoods are accumulated identically on both tracks.

    Args:
        prior_weight: 0 = no prior (blind), 1 = full prior injection
        prior_belief: Map's belief that obstacle exists (0, 1)

    Returns:
        Tuple of (decision, steps_taken, final_score)
    """
    prior_belief_clipped = np.clip(prior_belief, 1e-10, 1 - 1e-10)

    # CRITICAL FIX: use prior_belief_clipped on both sides (not prior_log_odds)
    prior_log_odds = np.log(prior_belief_clipped / (1 - prior_belief_clipped))

    # Blind track starts at 0; informed track starts at prior log-odds
    log_lr = prior_weight * prior_log_odds

    for i in range(min(max_steps, len(lidar_readings))):
        llr = calculate_sensor_log_lr(
            lidar_readings[i],
            camera_readings[i],
            p_h_lidar,
            p_not_h_lidar,
            p_h_camera,
            p_not_h_camera,
        )
        log_lr += llr

        if log_lr >= upper:
            return "accept", i + 1, log_lr
        if log_lr <= lower:
            return "reject", i + 1, log_lr

    # Budget exhausted: decide by sign of accumulated evidence
    decision = "accept" if log_lr >= 0 else "reject"
    return decision, max_steps, log_lr


# =============================================================================
# FEATURE COMPUTATION & GATING
# =============================================================================

def compute_divergence_sensor(prior_belief: float, sensor_mle: float) -> float:
    """D(x) = |prior_belief - sensor_MLE| clipped to [0, 1]."""
    return min(abs(prior_belief - sensor_mle), 1.0)


def compute_stakes_sensor(alpha: float, beta: float) -> float:
    """S(x) = |log(alpha / beta)|."""
    if alpha <= 0 or beta <= 0:
        return 0.0
    return abs(np.log(alpha / beta))


def gate_logistic_sensor(
    divergence: float,
    stakes: float,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 0.0
) -> float:
    """Logistic gate: w = sigma(-a*D + b*S + c). Trust decreases with divergence."""
    z = -a * divergence + b * stakes + c
    z = np.clip(z, -500, 500)
    w = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(w, 1e-10, 1 - 1e-10))


# =============================================================================
# SINGLE TRIAL EXECUTION
# =============================================================================

def run_single_sensor_trial(
    config: SensorTrialConfig,
    bench_config: SensorBenchmarkConfig
) -> Dict:
    """Run one complete sensor fusion trial with dual-track SPRT + trust gating."""
    rng = np.random.default_rng(seed=config.random_seed)

    # Generate sensor readings
    lidar_readings, camera_readings = generate_sensor_readings(config, rng)

    # Sensor MLE: fraction of positive detections across both sensors
    fused_readings = np.concatenate([lidar_readings, camera_readings])
    sensor_mle = np.mean(fused_readings[:config.max_steps * 2])

    # Compute features
    divergence = compute_divergence_sensor(config.map_prior_belief, sensor_mle)
    stakes = compute_stakes_sensor(config.alpha, config.beta)
    w = gate_logistic_sensor(divergence, stakes)

    # SPRT boundaries
    upper = np.log((1 - config.beta) / config.alpha)
    lower = np.log(config.beta / (1 - config.alpha))

    # Sensor likelihoods under H and ¬H (SAME for both tracks)
    p_h_lidar = config.lidar_reliability
    p_not_h_lidar = 1.0 - config.lidar_reliability
    p_h_camera = config.camera_reliability
    p_not_h_camera = 1.0 - config.camera_reliability

    # Track 1: Prior-Blind SPRT
    decision_blind, steps_blind, score_blind = run_sensor_sprt(
        lidar_readings, camera_readings,
        p_h_lidar, p_not_h_lidar,
        p_h_camera, p_not_h_camera,
        upper, lower, config.max_steps,
        prior_weight=0.0, prior_belief=config.map_prior_belief
    )

    # Track 2: Prior-Informed SPRT
    decision_informed, steps_informed, score_informed = run_sensor_sprt(
        lidar_readings, camera_readings,
        p_h_lidar, p_not_h_lidar,
        p_h_camera, p_not_h_camera,
        upper, lower, config.max_steps,
        prior_weight=1.0, prior_belief=config.map_prior_belief
    )

    # Blended decision
    blended_score = w * score_informed + (1 - w) * score_blind
    blended_decision = "accept" if blended_score >= 0 else "reject"
    blended_steps = int(round(w * steps_informed + (1 - w) * steps_blind))

    # Correctness (accept = "obstacle present")
    correct_blind = (decision_blind == "accept") == config.obstacle_present
    correct_informed = (decision_informed == "accept") == config.obstacle_present
    correct_blended = (blended_decision == "accept") == config.obstacle_present

    return {
        "domain": "B",
        "divergence": divergence,
        "stakes": stakes,
        "w_gate": w,
        "h_is_true": int(config.obstacle_present),
        "map_prior_belief": config.map_prior_belief,
        "prior_quality": config.prior_quality,
        "lidar_reliability": config.lidar_reliability,
        "camera_reliability": config.camera_reliability,
        "sensor_mle": sensor_mle,
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

def run_sensor_benchmark(bench_config: SensorBenchmarkConfig = None) -> pd.DataFrame:
    """Run full sensor fusion benchmark with randomized per-trial configs."""
    if bench_config is None:
        bench_config = SensorBenchmarkConfig()

    # Warn if kitti mode without path
    if bench_config.data_mode == "kitti" and not bench_config.kitti_path:
        print("WARNING: --data-mode=kitti but --kitti-path not set. Falling back to synthetic mode.")
        bench_config.data_mode = "synthetic"

    # Try real KITTI data first
    kitti_df = None
    if bench_config.data_mode == "kitti" and bench_config.kitti_path:
        print("Attempting to load KITTI data...")
        kitti_df = load_kitti_data(bench_config.kitti_path)
        if kitti_df is None:
            print("  Falling back to synthetic data mode.")
            bench_config.data_mode = "synthetic"

    master_rng = np.random.default_rng(seed=bench_config.random_seed)

    mode_str = "KITTI" if kitti_df is not None else "Synthetic"
    print(f"Running {bench_config.n_trials} sensor fusion trials ({mode_str} mode)...")
    print(f"  Obstacle probability: {bench_config.obstacle_prob}")
    print(f"  Readings per trial: {bench_config.readings_per_trial}")
    print(f"  LiDAR reliability: {bench_config.lidar_reliability_range}")
    print(f"  Camera reliability: {bench_config.camera_reliability_range}")
    print(f"  Prior quality — aligned: {bench_config.aligned_prob:.0%}, "
          f"stale: {bench_config.stale_prob:.0%}, "
          f"adversarial: {bench_config.adversarial_prob:.0%}")
    print()

    results = []

    if kitti_df is not None:
        n_available = len(kitti_df)
        n_trials = min(bench_config.n_trials, n_available)

        for i in range(n_trials):
            row = kitti_df.iloc[i]
            obstacle = bool(row["obstacle_detected"])
            lidar_conf = float(row["lidar_conf"])
            camera_conf = float(row["camera_conf"])

            trial_config = SensorTrialConfig(
                obstacle_present=obstacle,
                map_prior_belief=np.clip(lidar_conf + master_rng.normal(0, 0.1), 0.01, 0.99),
                lidar_reliability=max(lidar_conf, 0.5),
                camera_reliability=max(camera_conf, 0.5),
                prior_quality="aligned",
                alpha=master_rng.uniform(*bench_config.sprt_alpha_range),
                beta=master_rng.uniform(*bench_config.sprt_beta_range),
                max_steps=bench_config.readings_per_trial,
                random_seed=bench_config.random_seed + i,
            )

            result = run_single_sensor_trial(trial_config, bench_config)
            result["trial_id"] = i
            results.append(result)
    else:
        for i in range(bench_config.n_trials):
            trial_config = sample_sensor_trial_config(bench_config, master_rng, i)
            result = run_single_sensor_trial(trial_config, bench_config)
            result["trial_id"] = i
            results.append(result)

    df = pd.DataFrame(results)

    print(f"\nBenchmark complete. {len(df)} trials generated.")
    print("\nClass balance:")
    n_obstacle = int(df["h_is_true"].sum())
    n_no_obstacle = int((df["h_is_true"] == 0).sum())
    print(f"  Obstacle present: {n_obstacle} ({df['h_is_true'].mean():.1%})")
    print(f"  No obstacle:      {n_no_obstacle} ({(df['h_is_true'] == 0).mean():.1%})")

    if "prior_quality" in df.columns:
        print("\nPrior quality distribution:")
        for pq in ["aligned", "stale", "adversarial"]:
            count = int((df["prior_quality"] == pq).sum())
            print(f"  {pq}: {count} ({count/len(df):.1%})")

    print("\nAccuracy comparison:")
    print(f"  Prior-blind:    {df['correct_blind'].mean():.3f}")
    print(f"  Prior-informed: {df['correct_informed'].mean():.3f}")
    print(f"  Blended (gate): {df['correct_blended'].mean():.3f}")

    print("\nFeature ranges:")
    print(f"  Divergence: [{df['divergence'].min():.3f}, {df['divergence'].max():.3f}]")
    print(f"  Stakes:     [{df['stakes'].min():.3f}, {df['stakes'].max():.3f}]")
    print(f"  w (gate):   [{df['w_gate'].min():.3f}, {df['w_gate'].max():.3f}]")

    return df


def split_data(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test with seeded reproducibility."""
    rng = np.random.RandomState(random_seed)
    indices = rng.permutation(len(df))

    n_train = int(train_frac * len(df))
    n_val = int(val_frac * len(df))

    return (
        df.iloc[indices[:n_train]].reset_index(drop=True),
        df.iloc[indices[n_train:n_train + n_val]].reset_index(drop=True),
        df.iloc[indices[n_train + n_val:]].reset_index(drop=True),
    )


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain B: Sensor Fusion Benchmark (TGAI)")
    parser.add_argument("--n-trials", type=int, default=1000, help="Number of trials")
    parser.add_argument("--readings-per-trial", type=int, default=50, help="Sensor readings per trial")
    parser.add_argument("--obstacle-prob", type=float, default=0.55, help="Probability of obstacle per trial")
    parser.add_argument("--aligned-prob", type=float, default=0.5, help="Probability of aligned prior")
    parser.add_argument("--stale-prob", type=float, default=0.3, help="Probability of stale prior")
    parser.add_argument("--adversarial-prob", type=float, default=0.2, help="Probability of adversarial prior")
    parser.add_argument("--alpha-range", type=float, nargs=2, default=(0.01, 0.1),
                        metavar=("LOW", "HIGH"),
                        help="Range to sample per-trial SPRT alpha from")
    parser.add_argument("--beta-range", type=float, nargs=2, default=(0.01, 0.1),
                        metavar=("LOW", "HIGH"),
                        help="Range to sample per-trial SPRT beta from")
    parser.add_argument("--data-mode", type=str, default="synthetic", choices=["synthetic", "kitti"],
                        help="Data mode: synthetic or real KITTI")
    parser.add_argument("--kitti-path", type=str, default=None,
                        help="Path to KITTI dataset (if using kitti mode)")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--output", type=str, default="results/domain_b_data.csv", help="Output CSV path")

    args = parser.parse_args()

    # Validate probabilities
    total = args.aligned_prob + args.stale_prob + args.adversarial_prob
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Prior quality probabilities must sum to 1.0, got {total:.3f}")

    
    config = SensorBenchmarkConfig(
    n_trials=args.n_trials,
    readings_per_trial=args.readings_per_trial,
    random_seed=args.seed,
    obstacle_prob=args.obstacle_prob,
    aligned_prob=args.aligned_prob,
    stale_prob=args.stale_prob,
    adversarial_prob=args.adversarial_prob,
    sprt_alpha_range=tuple(args.alpha_range),
    sprt_beta_range=tuple(args.beta_range),
    data_mode=args.data_mode,
    kitti_path=args.kitti_path,
)
