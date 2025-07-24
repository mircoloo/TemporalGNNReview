#!/bin/bash

# Check input
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <market> <model>"
    exit 1
fi

market=$1
model=$2

# Directory for generated slurm scripts and logs
mkdir -p code/sbatch_outputs
mkdir -p code/sbatch_scripts

# Name of the SLURM job script to generate
slurm_file="code/sbatch_scripts/${market}_${model}.slurm"

# Generate the slurm script
cat <<EOF > "$slurm_file"
#!/bin/bash -l

#SBATCH --job-name=${market}_${model}_train
#SBATCH --output=code/sbatch_outputs/${market}_${model}.out
#SBATCH --error=code/sbatch_outputs/${market}_${model}.err
#SBATCH --mail-user=mirco.bisoffi@studenti.unipd.it
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=14-00:00:00
#SBATCH --gres=gpu

echo "Job started on \$(date)"
echo "Running on nodes: \${SLURM_NODELIST}"
echo "Market: ${market}"
echo "Model: ${model}"

conda activate gpu_env
python /home/mbisoffi/tests/TemporalGNNReview/code/run.py --market ${market} --model ${model}
EOF

# Submit the job
sbatch "$slurm_file"
