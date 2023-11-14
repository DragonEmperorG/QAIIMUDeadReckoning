#!/bin/bash

#SBATCH --partition=a100x4     #申请分区 `hpxg` 的计算资源

#SBATCH --ntasks-per-node=1  #申请每个节点上分配一个任务(进程)
#SBATCH --time=06:00:00      #计划最多运行 6 小时

#SBATCH --gres=gpu:1

#SBATCH --job-name=vdr-track-

#SBATCH --nodes=1

#SBATCH --partition=a100x4     #申请分区 `hpxg` 的计算资源

cd $SLURM_SUBMIT_DIR

R --vanilla CMD BATCH infile.R