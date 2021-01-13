#!/bin/bash
#set a job name
#SBATCH --job-name=AutoTracing
############
# a file for job output, you can check job progress
#SBATCH --output=out
############
# a file for errors from the job
#SBATCH --error=err
############
# number of tasks you are requesting
#SBATCH --mem=100G
############
# partition to use
#SBATCH --partition=gpu --gres=gpu:1
############
# number of nodes to distribute n tasks across
#SBATCH -N 1
############
# number of nodes to distribute n tasks across
#SBATCH --nodes 1 /bin/bash
############

module load python/3.6.6
conda activate atenv


$work=/home/kahaki/Python/Discovery
$cd $work


