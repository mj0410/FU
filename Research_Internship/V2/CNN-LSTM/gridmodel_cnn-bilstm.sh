#!/bin/bash

#SBATCH --job-name=cbi_gridmodel          
#SBATCH --mail-user=miniej94@zedat.fu-berlin.de  
#SBATCH --mail-type=end
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00 
#SBATCH --qos=standard

module add Python
module add CUDA/10.2.89-GCC-8.3.0
module add cuDNN/7.6.4.38-gcccuda-2019b

pip install tensorflow-gpu

cd /scratch/miniej94

python gridmodel_cnn-bilstm.py
