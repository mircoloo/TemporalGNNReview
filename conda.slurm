#!/bin/bash -l
### job name
#SBATCH --job-name=stock-data-creation

### 
#SBATCH --mail-user=mirco.bisoffi@studenti.unipd.it

### Standard output and standard error for our job
#SBATCH --error=result.err
#SBATCH --output=result.out

### queue/partition choosed
#SBATCH --partition=testing

### Number of tasks
#SBATCH --ntasks=1

### RAM requirement
#SBATCH --mem=8G

### Time limit for our job (ten minutes here: HH:MM:SS)
#SBATCH --time=14-00:00:00

### GPU request
#SBATCH --gres=gpu
#SBATCH --constraint=A6000

### Some useful informative commands
echo -n 'Date: '
date
echo -n 'Directory: '
pwd
echo -n 'This job will be executed on th following nodes: '
echo ${SLURM_NODELIST}
echo

SHELL=/bin/bash

### Jobs execution commands
conda info --envs
conda activate gpu_env
which python
python --version

python /home/mbisoffi/tests/TemporalGNNReview/code/main.py
