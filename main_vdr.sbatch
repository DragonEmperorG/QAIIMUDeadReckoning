#!/bin/bash

#SBATCH --job-name=vdr-track-0011

#SBATCH --time=1:00:00

#SBATCH --partition=hpxg     #申请分区 `hpxg` 的计算资源

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

source ~/.bashrc
source activate /home/qianlong/project/anaconda3/envs/pytorch

cd $SLURM_SUBMIT_DIR

python main_vdr.py