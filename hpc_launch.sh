#!/bin/bash
#
#
#SBATCH --partition=gpu
#SBATCH --account=EMAT017822
#SBATCH --error=hpc_dreamer3.err
#SBATCH --output=hpc_dreamer3.out
#SBATCH --mem=32G
#SBATCH --gpus=2

module load git/2.45.1-pqk5
module load libs/open3d/0.19.0

cd ~/hpc_dreamer3/ai2_thor_model_training_src/
pip install -e .

module load libs/open3d/0.19.0

cd ~/hpc_dreamer3/ai2_thor_model_training_src/thortils/
pip install .
cd ../../
python dreamerv3/main.py --configs indoorsae --batch_size 2 --run.envs 2
