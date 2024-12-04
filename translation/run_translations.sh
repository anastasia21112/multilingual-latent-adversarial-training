#!/bin/bash

#SBATCH --job-name="translate_ultrachat"
#SBATCH --output=output.txt
#SBATCH --time=3-17:00 # must set end time, default 2 days
#SBATCH -p healthyml # this line must not be changed 
#SBATCH -q healthyml-main # this line must not be changed 
#SBATCH --nodelist matcha
#SBATCH --gres=gpu:1 # allocate one gpu
#SBATCH -c 16 #16 cpus
#SBATCH --mem 100gb

source .mulbere-scratch/bin/activate #change this to your virtual env 

python --version
which python

python translate.py # you can pass args here like python script.py --arg1 "cat" --arg2 3.0