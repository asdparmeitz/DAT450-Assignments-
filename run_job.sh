#!/bin/bash
#SBATCH --job-name=A1_skeleton_job
#SBATCH --output=logs/%x-%j.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L4:1        # request two L4 GPU
#SBATCH --mem=8G
#SBATCH -p long


source ~/A1_project/.venv/bin/activate
cd ~/A1_project
python3 A1_skeleton.py
