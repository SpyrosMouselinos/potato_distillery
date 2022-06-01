#!/usr/bin/env bash
#SBATCH --job-name=distribute_test
#SBATCH --partition=common
#SBATCH --qos=4gpu7d
#SBATCH --output=./distribute_test.log
#SBATCH --gres=gpu:2
#SBATCH --nodelist=asusgpu4
#SBATCH --time=15
#SBATCH --mem=60G


echo "Testing..."
deepspeed --num_gpus 2 distribute.py 
echo "Testing Ended..."
