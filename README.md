# SlotFiLM: Condition-Adaptive Slot Mixing and Mode-Wise Modulation for Neural Operators

Official implementation of **SlotFiLM**, an advanced conditional Fourier Neural Operator architecture for multi-frame fluid dynamics forecasting and flow matching.

---

## 📌 Overview

Data-driven surrogates for turbulent flows increasingly rely on neural operators. When used as the conditional vector field in flow matching for multi-frame forecasting, operators need to coordinate spatial dynamics, temporal dependencies, and physical conditions. 

**SlotFiLM** resolves the representation paradox of temporal frames by:
1. **Explicit Temporal Slots**: Preserving an explicit `[B, S, C, H, W]` slot axis with condition-adaptive block-triangular slot mixing $T(c)$, enforcing directional context-to-forecast dependency.
2. **Bounded Mode-Wise FiLM**: Modulating truncated Fourier frequencies via low-rank, $\tanh$-bounded spectral gains to maintain numerical stability and incompressibility.

---

## 📂 Project Structure

```text
.
├── models/             # Core neural operator models
├── src/                # Metrics and common utilities
├── scripts/            # Training and inference pipeline
│   ├── runners/        # Shell scripts for training and evaluations
│   └── analysis/       # Diagnostic probes
└── utils.py            # Some training utils
```

---

## 🚀 Quick Start

### 1. Requirements

- Linux (Ubuntu 22.04 recommended)
- Python == 3.10, PyTorch == 2.1.2
- CUDA compatible GPU

### 2. Training

```bash
# Train SlotFiLM with Flow Matching on 2D Kolmogorov turbulence
bash scripts/runners/train_slot_family.sh
```

### 3. Evaluation

```bash
# Evaluate trained checkpoint
bash scripts/runners/eval_family.sh
```

---

## 📜 Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{wang2026slotfilm,
  title={SlotFiLM: Condition-Adaptive Slot Mixing and Mode-Wise Modulation for Neural Operators},
  author={Yuhao, Wang and Zhongxuan, Cai and Yinuo, Liu and Tianjian, Jiang and Yichen, Wang and Yuanxi, Peng},
  year={2026}
}
```
