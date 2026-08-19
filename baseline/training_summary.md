# Unitree R1 – Baseline Walking Policy (No Domain Randomization)

This document summarizes the first successful training of a Unitree R1 humanoid walking in the MuJoCo simulator using the official `unitree_rl_mjlab` framework.

**No domain randomization or sensor noise was applied** – this is a pure simulation baseline on flat terrain.

---

## 🎯 Objective

- Train a walking gait that tracks linear and angular velocity commands.
- Maximize upright time (episode length).
- Establish a baseline for later Sim‑to‑Real experiments.

---

## 🖥️ Training Setup

| Component | Specification |
|-----------|---------------|
| **Framework** | [`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab) |
| **Simulator** | MuJoCo |
| **RL Algorithm** | PPO (Proximal Policy Optimization) |
| **GPU** | NVIDIA L40S (47 GB VRAM) |
| **Parallel Environments** | 4,096 |
| **Total Iterations** | 10,000 |
| **Training Time** | ~2.5 hours |
| **Simulation Speed** | ~60,000 steps/second |

---

## 🚀 Training Command

```bash
python scripts/train.py Unitree-R1-Flat --env.scene.num-envs=4096
