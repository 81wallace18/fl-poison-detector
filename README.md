# DODGE: Defense Orchestrator for Detecting and Guarding against model-update Exploits

![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Security-blue)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

**DODGE** is an advanced Multi-Layer Perceptron (MLP) based defense mechanism designed to detect and eliminate malicious client updates before global aggregation in Federated Learning (FL) scenarios. 

This repository contains the official implementation of the paper: **"Defense Orchestrator for Detecting and Guarding against model-update Exploits in Federated Learning Scenarios"** (*Esteves et al., 2026*).

## 🛡️ About DODGE

In Federated Learning, malicious clients can manipulate their local data or model parameters to submit poisoned updates that compromise the global model. To counteract this, **DODGE** introduces a highly efficient, two-head MLP detector. 

DODGE extracts **210 statistical, spectral, spatial, and contextual features** from client models during the FL process (such as L2 norm, kurtosis, entropy, singular values, and FFT coefficients). It then utilizes:
1. **Binary Head:** Classifies client models as benign or malicious using cross-entropy loss.
2. **Label-Flipping Head:** Identifies complex label-flipping behaviors using focal loss and quantile Transformer threshold calibration.

### Supported Attacks
DODGE is evaluated against a highly heterogeneous threat model where 30% of clients act maliciously. It simultaneously defends against five distinct poisoning attacks:
- **Zero-Parameters:** Cancels learned weights to suppress training.
- **Shuffled-Layer Parameters:** Misaligns features via permutation.
- **Random Values:** Injects Gaussian noise to weaken the honest training signal.
- **Label Flipping:** Introduces systematic bias by shifting decision boundaries.
- **ALIE (A Little Is Enough):** Exploits natural variance to bypass distance-based filters.

## 📊 Evaluation & Results

Extensive simulations on the CIFAR-10 dataset using CNN architectures prove that DODGE significantly outperforms state-of-the-art defenses (zPROBE, MONZA, FedSIGN).

- **High Accuracy:** Reaches ~39% accuracy in environments severely compromised by 30% simultaneous attackers (outperforming zPROBE's 37% and MONZA's 31%).
- **Superior Recall:** Achieves consistent 83% - 90% recall rates.
- **Low Rejection Rates:** Successfully isolates malicious updates while minimizing the False Rejection Rate (FRR) to 10%-18% and the False Positive Rate (FPR) to ~8%-15%.

---

## 🚀 Quick Start

This project uses `uv` and requires Python 3.12.

### Installation
```bash
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  -r requirements.txt
bash scripts/check_project.sh
```

### Running Simulations
To perform a dry-run of the CIFAR-10 configuration without altering files:
```bash
bash scripts/run_full.sh cifar10 --dry-run
```

To initiate the full DODGE defense experiment (10 executions, 300 global rounds):
```bash
GLOBAL_ROUNDS=300 TIMES=10 bash scripts/run_full.sh cifar10 --background
```

A full step-by-step guide is available in [HOWTO.md](docs/guides/HOWTO.md).

## 📁 Repository Structure

```text
.
├── docs/                    # Active guides, experimental results, and methodology history
├── notebooks/               # Reproducible Jupyter notebooks for data analysis
├── src/                     # DODGE core: training, inference, and feature extraction
├── scripts/                 # Bash workflows and pipeline automation
├── artifacts/               # Local results and generated graphs (ignored by Git)
└── PFLlibMonza/             # Integrated Federated Learning simulator
```

*Note: The core modules used by the framework (`cc_mlp.py`, `context_features.py`, `features.py`, `fl_save.py`) are duplicated in `src/` and `PFLlibMonza/system/flcore/detector/`. After altering them in `src/`, synchronize the runtime by executing:*
```bash
python3 scripts/_check_runtime_sync.py
```

## 📝 Outputs

| Directory | Content |
|---|---|
| `artifacts/runs/<dataset>/<run-id>/` | Logs, CSVs, PDF graphs, and executed notebooks. |
| `artifacts/models/<dataset>/mlp/` | Trained models, scalers, thresholds, and diagnostics. |
| `artifacts/dumps/<dataset>/` | Temporary state dumps used during detector training. |

## 📚 Documentation
- [HOWTO.md](docs/guides/HOWTO.md): Step-by-step installation and execution guide.
- [MONZA_RESULTS.md](docs/results/MONZA_RESULTS.md): Legacy experimental results.
- [EVOLUTION.md](docs/history/EVOLUTION.md): Architectural decisions and methodological evolution.
- [scripts/README.md](scripts/README.md): Reference for the active pipeline scripts.

## 📝 Citation
If you utilize DODGE in your research, please cite our work:
```bibtex
@article{esteves2026dodge,
  title={Defense Orchestrator for Detecting and Guarding against model-update Exploits in Federated Learning Scenarios},
  author={Esteves, Anderson and Gonçalves, João and Veiga, Rafael and Rosário, Denis and Cerqueira, Eduardo},
  journal={Journal of Internet Services and Applications},
  year={2026}
}
```
