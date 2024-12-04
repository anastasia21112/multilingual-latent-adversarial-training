#!/bin/bash

# TODO add output logs, add setup bash to run script thing

#SBATCH --job-name="get_resps"
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --time=2-00:00 # must set end time, default 2 days
#SBATCH -p healthyml
#SBATCH -q healthyml-main
#SBATCH --gres=gpu:1 # allocate one gpu
#SBATCH -c 16 #16 cpus
#SBATCH --mem 100gb
#SBATCH --nodelist espresso
#SBATCH --nodes 1

source activate ../anaconda3/envs/multilingual_tlat


python --version
which python


export WANDB_MODE=disabled

# arg1_values=('Accountant' 'Astronaut' 'Biologist' 'Carpenter' 'Civil Engineer' 'Clerk' 'Detective' 'Editor' 'Firefighter' 'Interpreter' 'Manager' 'Nutritionist' 'Paramedic' 'Pharmacist' 'Physicist' 'Pilot' 'Reporter' 'Security Guard' 'Scientist' 'Web Developer')
# arg1_values=('es' 'zh' 'ko' 'ar' 'el' 'sw' 'am' 'en')
# arg2_values=('es' 'zh' 'ko' 'ar' 'el' 'sw' 'am' 'en')

arg1_values=('en' 'vi')
arg2_values=('en' 'vi')
# arg1_value=${arg1_values[$SLURM_ARRAY_TASK_ID-1]}

pip install -r requirements.txt
bash install_tasks_from_github.sh

cd notebooks
python jailbreaks_script.py --languages_train en es ko ar sw am --languages_sft en --sequential

# python notebooks/jailbreaks_script.py