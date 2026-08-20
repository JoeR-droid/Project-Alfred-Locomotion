# Project Alfred: Locomotion  
**Unitree R1 – From Flat‑Ground Baseline to Domain‑Randomized Robustness**

---

## The Story

This project is a two‑phase journey to make a Unitree R1 humanoid walk not just in simulation, but in the real world.

**Phase 1** was the baseline: I took the official `unitree_rl_mjlab` framework, ran the standard flat‑terrain training on an L40S, and got a reliable walking policy.

**Phase 2** pushed further: I added domain randomization (friction, mass, COM offset, sensor noise) and fine‑tuned the baseline policy on an A100, turning a perfect‑simulation walker into one that can handle real‑world variation.

The policy is now exported to ONNX and ready for the physical robot – but **real‑hardware deployment is the ongoing phase**, and that's the next milestone.

---

## Phase 1: Baseline (No DR)

I started from scratch with the official flat‑terrain config.

| **Hardware** | NVIDIA L40S (48 GB VRAM) |
| :--- | :--- |
| **Environments** | 4096 parallel sims |
| **Terrain** | Flat plane |
| **Domain Randomization** | None |
| **Training Command** | `python scripts/train.py Unitree-R1-Flat --env.scene.num-envs=4096` |
| **Final Reward** | ~30–40 |
| **Episode Length** | 1000 (max) – never falls |
| **Fall Rate** | < 5% |
| **Training Time** | ~2.5 hours |

The baseline policy learned to walk forward, turn, and recover from small disturbances – a solid starting point.

---

## Phase 2: Domain Randomization (DR)

I loaded the **best checkpoint** from Phase 1 and fine‑tuned it on a fresh A100 instance, modifying the environment to inject randomness.

| **Hardware** | NVIDIA A100 (40 GB VRAM) |
| :--- | :--- |
| **Environments** | 8192 parallel sims |
| **Terrain** | Flat plane (forced) |
| **Training Command** | `python scripts/train.py Unitree-R1-Rough --env.scene.num-envs=8192 --agent.load-run ... --agent.load-checkpoint model_6800.pt` |
| **Final Reward** | ~58 |
| **Episode Length** | 1000 (max) – zero falls |
| **Fall Rate** | 0% |
| **Training Time** | ~1 hour (resumed from checkpoint) |

I overrode the `rough` config to keep the terrain flat while injecting DR. The key modifications:

### Domain Randomization Parameters

| Parameter | Range | Mode |
| :--- | :--- | :--- |
| Foot friction | 0.2 – 1.5 (was 0.3–1.6) | Startup |
| COM offset (x, y, z) | ±0.07 m (was ±0.05) | Startup |
| Body mass | 0.8 – 1.2× (scaled) | Startup |
| Joint encoder bias | ±0.015 rad | Startup |
| Random pushes | x/y/z/roll/pitch/yaw | Interval (5–6 s) |
| Sensor noise (Gaussian) | 0.02 – 0.1 std | Startup |

I also **narrowed the lateral command range** (`lin_vel_y = -0.3..0.3`) and **increased the forward reward weight** from 1.0 to 2.0 to fix an initial side‑walking issue.

---

## Results Comparison

| Metric | Baseline (No DR) | DR Fine‑Tuned |
| :--- | :--- | :--- |
| Mean reward | 30–40 | **58** |
| Episode length | 1000 | 1000 |
| Falls | <5% | **0%** |
| Forward tracking | moderate | **high (1.7/2.0)** |
| Robustness to friction changes | none | **0.2–1.5** |
| Sensor noise tolerance | none | **Gaussian** |

The DR policy walks forward stably, handles variations in ground friction, mass shifts, and noisy sensors – exactly what you need for the real world.

---

## Key Code Modifications

The core change lives in `config/env_cfgs.py` – I modified the `unitree_r1_rough_env_cfg` function to:

- Force terrain to `"plane"`.
- Override friction, COM, and mass ranges.
- Inject Gaussian sensor noise.
- Narrow lateral commands and boost forward reward weight.

Everything else stays within the official `unitree_rl_mjlab` framework.

---

## Deployment & Ongoing Work

The final policy is exported as **ONNX** (`policy.onnx` and `policy.onnx.data`) and is ready for the physical Unitree R1 using the built‑in C++ deployment pipeline and will tested once a Unitree R1 comes into my possesion:

```bash
cd deploy/robots/r1
mkdir build && cd build
cmake .. && make
./r1_ctrl --network=enp5s0      # real robot
# or
./r1_ctrl --network=lo          # simulation test
