#!/bin/bash
# Usage: ./render_video.sh /path/to/checkpoint.pt
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi
MUJOCO_GL=egl python scripts/play.py Unitree-R1-Rough --checkpoint-file $1 --video True --video-length 500
