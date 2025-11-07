#!/usr/bin/env zsh

#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=dependent_async.output

cd $SLURM_SUMIT_DIR
module load gcc
g++ -std=c++20 -O2 -Wall dependent_async.cpp -o dependent_async -I/path/to/taskflow -pthread
./dependent_async