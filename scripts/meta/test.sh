#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition bosch_cpu-cascadelake    # short: -p bosch_cpu-cascadelake

# Define a name for your job
#SBATCH --job-name hposuite_env_test            #  short: -J hposuite_env_test

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/%x-%A_meta.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A_meta.out
#SBATCH --error logs/%x-%A_meta.err    # STDERR  short: -e logs/%x-%A_meta.out

# Since using CPU Partion, define the number of cpus required per node
#SBATCH --cpus-per-task 30

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
source ~/repos/envs/hposuite_env/bin/activate

# Running the job
start=`date +%s`

python -m hposuite --optimizer RandomSearch Optuna --benchmark pymoo-ackley bbob-20-2-1 --budget 10 --num_seeds 3 --overwrite -study_name env_test --auto_env_handling

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime