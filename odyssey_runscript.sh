#!/bin/bash


#SBATCH -J fitsig
#SBATCH -n 8 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 12:00:00 # Runtime 
#SBATCH -p conroy # Partition to submit to
#SBATCH --mem=2000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -o /n/regal/conroy_lab/bdjohnson/starrv/logs/fitsig_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/regal/conroy_lab/bdjohnson/starrv/logs/fitsig_%A_%a.err # Standard err goes to this file

spj=2
ndatafiles=10
ncpu=$SLURM_JOB_CPUS_PER_NODE
niter=1024


cd /n/regal/conroy_lab/bdjohnson/starrv
source activate pro
echo "RUNNING JOB ARRAY ID: " $SLURM_ARRAY_TASK_ID
start=$((($SLURM_ARRAY_TASK_ID-1)*$spj))
stop=$(($SLURM_ARRAY_TASK_ID*$spj))
ind=$(($SLURM_ARRAY_TASK_ID%$ndatafiles+1))
libname="data/ckc_R10K_${ind}.h5"
echo $start
echo $stop
echo $ncpu
python fit_broad_lambda.py $start $stop $ncpu --niter=${niter} --libname=${libname} --verbose=False
