#!/bin/bash
#SBATCH --job-name=IM_(20)
#SBATCH --output=out1
#SBATCH --error=err1
#SBATCH --mem=100G
#SBATCH --partition=gpu --gres=gpu:p100:1
#SBATCH -N 1
#module load python/3.6.6
#conda activate atenv
module unload python/3.7.0
module load python/3.6.6
module load cuda/9.0
module list
ulimit -s unlimited
work=/home/kahaki/Python/Discovery
cd $work
python3.6 TrainScenariosAndImage_WorkinDec2019_batch_epochs.py 20
