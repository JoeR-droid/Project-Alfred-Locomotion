# Unitree R1 – Domain Randomization Fine‑Tuning

This phase takes the baseline checkpoint from Phase 1 and fine‑tunes it with domain randomization to bridge the sim‑to‑real gap.

---

## 🎯 Objective

- Add robustness to real‑world variations (friction, mass, COM, sensor noise).
- Fine‑tune the baseline policy on a more powerful GPU with more parallel environments.
- Export to ONNX for deployment on the physical Unitree R1.

---

## 🖥️ Hardware & Software

| Component | Details |
| :--- | :--- |
| GPU | NVIDIA A100 (40 GB VRAM) |
| Parallel Environments | 8,192 |
| RL Algorithm | PPO (Proximal Policy Optimization) |
| Simulation Engine | MuJoCo |
| Framework | `unitree_rl_mjlab` (Unitree official) |
| Training Duration | ~1 hour (resumed from checkpoint 6800) |
| Starting Checkpoint | `model_6800.pt` from Phase 1 |

---

## 🚀 Training Command

```bash
python scripts/train.py Unitree-R1-Rough \
    --env.scene.num-envs=8192 \
    --agent.load-run logs/rsl_rl/r1_velocity/2026-08-19_02-29-12 \
    --agent.load-checkpoint model_6800.pt \
    --agent.max-iterations=10000
