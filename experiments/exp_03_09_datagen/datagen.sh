#!/bin/bash
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

experiment="exp_02_27_mobilenetV2_bins_eo"
venv_dir="/home/${USER}/personality-machine/venv"
repo_dir="/home/${USER}/personality-machine/apr-model"
ckpt_dir="/rds/user/${USER}/hpc-work/personality-machine/experiments"
data_dir="/rds/user/${USER}/hpc-work/personality-machine/"

# === Run ===

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
module load python/3.8 cuda/11.2 cudnn/8.1_cuda-11.2
module load ffmpeg

source ${venv_dir}/bin/activate

cd /home/elyro2/personality-machine/apr-model/datasets/first_impressions
tfds build --config large --data_dir /rds/user/elyro2/hpc-work/personality-machine/tfds-large/ --manual_dir /rds/user/elyro2/hpc-work/personality-machine/first-impressions/ --overwrite