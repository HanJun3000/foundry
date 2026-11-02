#!/usr/bin/env bash
#SBATCH --job-name=rfo_cycle
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=108G
#SBATCH --time=3-00:00:00
#SBATCH --output=rfo_cycle_%j.log

# Adjust resources for your cluster. Configure model environments as described
# in docs/installation.md, then submit from models/rfo:
# sbatch src/rfo/cycling/run_cycle.slurm [optional RFO flags]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from models/rfo}"
./rfo --config src/rfo/cycling/configs/example.yaml "$@"
