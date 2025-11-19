#!/bin/bash

# Check input
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <market> <model> <normalization> <adjnorm> <adjnormthreshold> <run_id>"
    echo "Example: $0 nasdaq dgdnn zscore False 0.5 test1"
    exit 1
fi

market=$1
model=$2
normalization=$3
adjnorm=$4
adjnormthreshold=$5
run_id=$6

# Directory for generated slurm scripts and logs
mkdir -p code/sbatch_outputs
mkdir -p code/sbatch_scripts

# Name of the SLURM job script to generate
slurm_file="code/sbatch_scripts/${market}_${model}_${run_id}.slurm"

# Generate the slurm script
cat <<EOF > "$slurm_file"
#!/bin/bash -l

#SBATCH --job-name=${market}_${model}_${normalization}_train
#SBATCH --output=code/sbatch_outputs/${market}_${model}_${normalization}_${run_id}.out
#SBATCH --error=code/sbatch_outputs/${market}_${model}_${normalization}_${run_id}.err
#SBATCH --mail-user=mirco.bisoffi@studenti.unipd.it
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=14-00:00:00
##SBATCH --gres=gpu
#SBATCH --account=thesis

echo "Job started on \$(date)"
echo "Running on nodes: \${SLURM_NODELIST}"
echo "Market: ${market}"
echo "Model: ${model}"
echo "Norm: ${normalization}"
echo "Adj norm: ${adjnorm}"

conda activate gpu_env
python /home/mbisoffi/tests/TemporalGNNReview/code/run.py --market ${market} --model ${model} --norm ${normalization} --adjnorm ${adjnorm} --adjnorm_threshold ${adjnormthreshold}
EOF

# Submit the job
sbatch "$slurm_file"
