#!/bin/bash
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

# === Config ===

experiment="exp_02_20_baseline_resnet"
venv_dir="/home/${USER}/personality-machine/venv"
repo_dir="/home/${USER}/personality-machine/apr-model"
ckpt_dir="/rds/user/${USER}/hpc-work/personality-machine/experiments"
data_dir="/rds/user/${USER}/hpc-work/personality-machine/tfds"

num_epochs=500
save_every=0.5

# === Run ===

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
module load python/3.8 cuda/11.2 cudnn/8.1_cuda-11.2

source ${venv_dir}/bin/activate
cd ${repo_dir}

python run.py train \
--experiment ${experiment} \
--data_dir ${data_dir} \
--ckpt_base ${ckpt_dir} \
--num_epochs ${num_epochs} \
--save_every ${save_every} \