#!/bin/bash

# Phase 1: Baseline (No DR) – L40S, 4096 envs
echo "Phase 1 (Baseline, no DR):"
echo "python scripts/train.py Unitree-R1-Flat --env.scene.num-envs=4096"

echo ""

# Phase 2: DR Fine‑Tuning – A100, 8192 envs
echo "Phase 2 (DR Fine‑Tuning from baseline checkpoint):"
echo "python scripts/train.py Unitree-R1-Rough --env.scene.num-envs=8192 --agent.load-run logs/rsl_rl/r1_velocity/2026-08-19_02-29-12 --agent.load-checkpoint model_6800.pt --agent.max-iterations=10000"

echo ""

# Render video
echo "Render video (10 seconds):"
echo "MUJOCO_GL=egl python scripts/play.py Unitree-R1-Rough --checkpoint-file logs/rsl_rl/r1_velocity/2026-08-19_02-29-12/model_1200.pt --video True --video-length 500"
