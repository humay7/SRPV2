#!/bin/bash
#SBATCH --job-name=run_DA_exps
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=abbasli@master.ismll.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
source activate GPU
srun python evaluate.py
