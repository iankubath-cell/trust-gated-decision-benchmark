# Trust-Gated Decision Mechanisms Benchmark (TGAI-Bench)

**Empirical cross-domain benchmark for trust-gated auxiliary information in decision systems.**

---

## ⚠️ Important Disclaimer

This repository implements an **empirical benchmark**, not a novel theoretical framework. The underlying concept of trust-weighted information combination is well-established in existing literature (see "Background" below). Our contribution is the **cross-domain transfer test**, not the gating mechanism itself.

---

## What We're Testing

**Primary Question:**  
Can a trust-gating function $g(x)$ fitted on one decision domain transfer to structurally different domains with **only unit rescaling** (no functional reshaping)?

**Secondary Question:**  
What functional form ($\text{logistic}$, $\text{linear}$, $\text{threshold}$) best balances performance across heterogeneous domains?

---

## Background & Prior Art

This work builds upon established frameworks:

| Literature | Contribution | Citation |
|------------|--------------|----------|
| Power Priors | Historical data weighted by $\alpha \in [0,1]$ | Ibrahim & Chen, 2000 |
| SafeBayes | Learning rate $\eta$ chosen to minimize cumulative loss | Grünwald & van Ommen |
| Mixture-of-Experts | Gated routing to expert subnetworks | Jacobs et al., 1991 |
| Robust Statistics | $\epsilon$-contamination models for adversarial components | Huber, 1964 |
| Sensor Fusion | Adaptive Kalman filtering with reliability estimates | Welch & Bishop, 2006 |

**Our contribution:** Empirical instantiation and transfer testing across **three distinct domains** (sequential testing, sensor fusion, fraud detection), not discovery of the general principle.

---

## Research Domains

| Domain | Data Source | Ground Truth | Auxiliary Signal | Corruption Type |
|--------|-------------|--------------|------------------|-----------------|
| **A: Sequential Testing** | Synthetic | Coin bias (known) | Prior estimate | Prior staleness / drift |
| **B: Sensor Fusion** | KITTI Odometry Dataset | GPS/IMU trajectory | Wheel odometry | Gaussian sensor noise |
| **C: Fraud Detection** | IEEE-CIS Fraud Dataset | Transaction labels | User metadata | Adversarial framing |

At least two domains use **real, independently-collected datasets** to avoid "built-to-pass" criticism.

---

## Pre-Registration

- **Version:** 1.0 (locked before analysis)
- **Date:** July 29, 2026
- **Download:** [`PRE_REGISTRATION.pdf`](./PRE_REGISTRATION.pdf)

**Key Falsifiable Claim:**  
A logistic gating function $g(x) = \sigma(a \cdot D(x) - b \cdot S(x) - c)$ fitted on Domain A will calibrate on Domains B and C within **1.5× calibration error** of domain-specific refit (after only rescaling units).

**Failure Condition:**  
Transferred calibration error exceeds **2.0×** domain-specific refit → reject transfer hypothesis.

---

## Repository Structure
# Trust-Gated Decision Mechanisms Benchmark (TGAI-Bench)

**Empirical cross-domain benchmark for trust-gated auxiliary information in decision systems.**

---

## ⚠️ Important Disclaimer

This repository implements an **empirical benchmark**, not a novel theoretical framework. The underlying concept of trust-weighted information combination is well-established in existing literature (see "Background" below). Our contribution is the **cross-domain transfer test**, not the gating mechanism itself.

---

## What We're Testing

**Primary Question:**  
Can a trust-gating function $g(x)$ fitted on one decision domain transfer to structurally different domains with **only unit rescaling** (no functional reshaping)?

**Secondary Question:**  
What functional form ($\text{logistic}$, $\text{linear}$, $\text{threshold}$) best balances performance across heterogeneous domains?

---

## Background & Prior Art

This work builds upon established frameworks:

| Literature | Contribution | Citation |
|------------|--------------|----------|
| Power Priors | Historical data weighted by $\alpha \in [0,1]$ | Ibrahim & Chen, 2000 |
| SafeBayes | Learning rate $\eta$ chosen to minimize cumulative loss | Grünwald & van Ommen |
| Mixture-of-Experts | Gated routing to expert subnetworks | Jacobs et al., 1991 |
| Robust Statistics | $\epsilon$-contamination models for adversarial components | Huber, 1964 |
| Sensor Fusion | Adaptive Kalman filtering with reliability estimates | Welch & Bishop, 2006 |

**Our contribution:** Empirical instantiation and transfer testing across **three distinct domains** (sequential testing, sensor fusion, fraud detection), not discovery of the general principle.

---

## Research Domains

| Domain | Data Source | Ground Truth | Auxiliary Signal | Corruption Type |
|--------|-------------|--------------|------------------|-----------------|
| **A: Sequential Testing** | Synthetic | Coin bias (known) | Prior estimate | Prior staleness / drift |
| **B: Sensor Fusion** | KITTI Odometry Dataset | GPS/IMU trajectory | Wheel odometry | Gaussian sensor noise |
| **C: Fraud Detection** | IEEE-CIS Fraud Dataset | Transaction labels | User metadata | Adversarial framing |

At least two domains use **real, independently-collected datasets** to avoid "built-to-pass" criticism.

---

## Pre-Registration

- **Version:** 1.0 (locked before analysis)
- **Date:** July 29, 2026
- **Download:** [`PRE_REGISTRATION.pdf`](./PRE_REGISTRATION.pdf)

**Key Falsifiable Claim:**  
A logistic gating function $g(x) = \sigma(a \cdot D(x) - b \cdot S(x) - c)$ fitted on Domain A will calibrate on Domains B and C within **1.5× calibration error** of domain-specific refit (after only rescaling units).

**Failure Condition:**  
Transferred calibration error exceeds **2.0×** domain-specific refit → reject transfer hypothesis.

---

## Repository Structure
trust-gated-decision-benchmark/ ├── README.md # This file ├── PRE_REGISTRATION.pdf # Locked pre-registration document ├── LICENSE # MIT License ├── domains/ │ ├── sequential_testing.py # Synthetic Domain A generation │ ├── sensor_fusion.py # KITTI dataset loading + preprocessing │ └── fraud_detection.py # IEEE-CIS dataset loading + preprocessing ├── gating/ │ ├── g_function.py # Logistic/Linear/Threshold implementations │ └── calibrator.py # Parameter fitting routines ├── transfer_tests/ │ ├── cross_domain_eval.py # Main transfer protocol │ └── ablation_study.py # Compare functional forms ├── results/ │ └── [empty, populated post-analysis] └── scripts/ ├── run_benchmark.sh # Full pipeline orchestration └── generate_plots.py # Visualization utilities

---

## Installation

bash git clone https://github.com/IanKubath/trust-gated-decision-benchmark.git cd trust-gated-decision-benchmark
Create virtual environment

python -m venv venv source venv/bin/activate # Linux/macOS venv\Scripts\activate # Windows
Install dependencies

pip install -r requirements.txt

**Dependencies:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

---

## Quick Start

### 1. Generate / Load Domains

bash python -m domains.sequential_testing --num-samples 10000 python -m domains.sensor_fusion --dataset-path ./data/kitti python -m domains.fraud_detection --dataset-path ./data/ieee-cis

### 2. Train Gating Function on Domain A

bash python -m gating.calibrator --domain sequential_testing --form logistic

### 3. Run Transfer Tests

bash python -m transfer_tests.cross_domain_eval --trained-gate ./models/domain_A_gate.pkl

### 4. Ablation Study

bash python -m transfer_tests.ablation_study --compare-logistic-linear-threshold

---

## Success Criteria

| Metric | Success Threshold | Failure Threshold |
|--------|------------------|-------------------|
| **Calibration Error (ECE)** | Transferred ≤ 1.5× domain-specific refit | Transferred > 2.0× domain-specific refit |
| **AUC-ROC** | Transferred within ±0.05 of refit | Transferred > 0.10 below refit |
| **Brier Score** | Transferred ≤ 1.5× domain-specific refit | Transferred > 2.0× domain-specific refit |

**Decision Rule:** Two out of three metrics must meet success thresholds to claim transfer.

---

## Expected Outcomes & Publication Paths

| Outcome | Interpretation | Target Venue |
|---------|---------------|--------------|
| **Full Transfer** | TGAI is domain-agnostic | arXiv (cs.LG, stat.ML) |
| **Partial Transfer** | Works for noise, fails for adversarial | Workshop (NeurIPS/ICLR Safety Track) |
| **Complete Failure** | Binary gates required for unobservable signals | arXiv (cs.CR, AI safety community) |

**Note:** All three outcomes are publishable. Negative results are explicitly welcomed and protected by pre-registration.

---

## Reproduction & Transparency

- All code is open source (MIT License)
- Pre-registration locked before analysis
- No HARKing (Hypothesizing After Results are Known)
- Full data access instructions provided in each domain module
- Will release trained checkpoints and calibration curves post-analysis

---

## Citation

If you use this benchmark in your research:

bibtex @software{kubath2026tgai, author = {Kubath, Ian}, title = {Trust-Gated Decision Mechanisms Benchmark (TGAI-Bench)}, year = {2026}, url = {https://github.com/IanKubath/trust-gated-decision-benchmark}, license = {MIT} }

---

## Contact & Contributions

**Author:** Ian Kubath  
**Email:** ViraListen@proton.me  
**Issues:** Please report bugs or suggestions via GitHub Issues

Contributions welcome for:
- Additional domain implementations
- Alternative gating function forms
- Calibration visualization improvements

---

## License

MIT License. See [`LICENSE`](./LICENSE) for full terms.

---

## Related Work

This benchmark connects to broader literature on:

- **Robust Bayesian Inference:** Power priors, commensurate priors, discounting
- **Adversarial Machine Learning:** Attack vectors, robustness certification
- **AI Safety:** Content moderation, safety alignment, verification architectures
- **Statistical Decision Theory:** Sequential testing, Wald SPRT, model criticism
