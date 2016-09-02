#!/bin/bash
#
# SLURM resource specifications
# (use an extra '#' in front of SBATCH to comment-out any unused options)
#
#SBATCH --job-name=skel_NTASKS   # this will show up in the output of 'squeue'
#SBATCH --time=0-0:59:59       # specify the requested wall-time
#SBATCH --partition=astro_short  # specify the partition to run on
##SBATCH --nodes=4               # number of nodes allocated for this job
##SBATCH --ntasks-per-node=20    # number of MPI ranks per node
##SBATCH --cpus-per-task=1       # number of OpenMP threads per MPI rank
#SBATCH --ntasks=NTASKS
##SBATCH --exclude=<node list>   # exclude the nodes named in <node list> (e.g. --exclude=node786)
#SBATCH --mail-type=ALL

# Load default settings for environment variables
#source /users/software/astro/startup.d/modules.sh

# If required, replace specific modules
# module unload intelmpi
# module load mvapich2

# When compiling remember to use the same environment and modules
# SLURM support is automatically linked in if using Intel compiler v15+ and intelMPI (default)
# This requires the program to be compiled with intel compilers (mpiifort, mpiicc, mpiicpc)

# Execute the code
COMMAND
