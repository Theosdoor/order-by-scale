#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:pascal:1
#SBATCH --array=0-7
#SBATCH --output=$HOME/list-comp/logs/slurm-%A_%a.out

set -u

source /etc/profile
module load cuda/12.5

mkdir -p "$HOME/list-comp/logs"
mkdir -p "$HOME/list-comp/results"

cd "$HOME/list-comp"

stdbuf -oL $HOME/.venv/bin/python \
  --device cuda \
  --output "$HOME/list-comp/results/results.csv"
