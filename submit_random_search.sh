#!/bin/bash

# Check input
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <market> <model> <normalization> <n_trials> <epochs> <run_id>"
    echo "Example: $0 nasdaq dgdnn zscore 20 100 search1"
    exit 1
fi

market=$1
model=$2
normalization=$3
n_trials=$4
epochs=$5
run_id=$6

# Directory for generated slurm scripts and logs
mkdir -p code/sbatch_outputs
mkdir -p code/sbatch_scripts

# Name of the SLURM job script to generate
slurm_file="code/sbatch_scripts/search_${market}_${model}_${normalization}_${run_id}.slurm"

# Determine GPU requirements
gpu_option=""
if [ "$model" == "darnn" ] || [ "$model" == "hyperstockgat" ]; then
    gpu_option="#SBATCH --gres=gpu:1"
fi

# Generate the slurm script
cat <<EOF > "$slurm_file"
#!/bin/bash -l

#SBATCH --job-name=search_${market}_${model}_${normalization}_${run_id}
#SBATCH --output=code/sbatch_outputs/search_${market}_${model}_${normalization}_${run_id}.out
#SBATCH --error=code/sbatch_outputs/search_${market}_${model}_${normalization}_${run_id}.err
#SBATCH --mail-user=mirco.bisoffi@studenti.unipd.it
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
${gpu_option}
#SBATCH --account=thesis

echo "Job started on \$(date)"
echo "Running on nodes: \${SLURM_NODELIST}"
echo "Market: ${market}"
echo "Model: ${model}"
echo "Normalization: ${normalization}"
echo "Trials: ${n_trials}"
echo "Epochs: ${epochs}"

conda activate gpu_env
python /home/mbisoffi/tests/TemporalGNNReview/code/random_search.py --market ${market} --model ${model} --norm ${normalization} --n_trials ${n_trials} --epochs ${epochs}
EOF

# Submit the job
sbatch "$slurm_file"
