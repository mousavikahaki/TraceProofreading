#!/bin/bash

for ID in $(seq 1 1 16)
do
                echo "#!/bin/bash">subjob1.bash
                echo '#SBATCH --job-name=IM_('$ID')'>>subjob1.bash
                echo "#SBATCH --output=out1" >> subjob1.bash
                echo "#SBATCH --error=err1">>subjob1.bash
                echo "#SBATCH --mem=100G">>subjob1.bash
				echo "#SBATCH --partition=gpu --gres=gpu:p100:1">>subjob1.bash
                echo "#SBATCH -N 1">>subjob1.bash
				echo "#module load python/3.6.6">>subjob1.bash
				echo "#conda activate atenv">>subjob1.bash
				echo "module unload python/3.7.0">>subjob1.bash
				echo "module load python/3.6.6">>subjob1.bash
				echo "module load cuda/9.0">>subjob1.bash
				echo "module list">>subjob1.bash
				echo "ulimit -s unlimited">>subjob1.bash
                echo "work=/home/kahaki/Python/Discovery" >> subjob1.bash
                echo 'cd $work' >> subjob1.bash
                echo 'python3.6 TrainingDiscovery.py '$ID'' >>subjob1.bash
                
                sbatch subjob1.bash
done
