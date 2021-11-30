#!/bin/bash

#SBATCH --job-name=grid_cnn-lstm 
#SBATCH --mail-user=miniej94@zedat.fu-berlin.de
#SBATCH --mail-type=end
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00 
#SBATCH --qos=prio

module add Python
module add CUDA/10.2.89-GCC-8.3.0
module add cuDNN/7.6.4.38-gcccuda-2019b 

pip install tensorflow-gpu

cd /scratch/miniej94

python grid_cnn-lstm.py
