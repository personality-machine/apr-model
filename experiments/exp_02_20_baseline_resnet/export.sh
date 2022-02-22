#!/bin/bash

# === Config ===

experiment="exp_02_20_baseline_resnet"
venv_dir="/home/${USER}/personality-machine/venv"
repo_dir="/home/${USER}/personality-machine/apr-model"
ckpt_dir="/rds/user/${USER}/hpc-work/personality-machine/experiments"
data_dir="/rds/user/${USER}/hpc-work/personality-machine/tfds"

saved_model_name="saved_model"
ckpt=1

# === Run ===

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
module load python/3.8 cuda/11.2 cudnn/8.1_cuda-11.2

source ${venv_dir}/bin/activate
cd ${repo_dir}

python run.py export \
--experiment ${experiment} \
--ckpt_base ${ckpt_dir} \
--ckpt ${ckpt} \
--saved_model_path ${repo_dir}/experiments/${experiment}/${saved_model_name}