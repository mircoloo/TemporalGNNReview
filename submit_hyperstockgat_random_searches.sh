#!/bin/bash

# Configuration
N_TRIALS=20
EPOCHS=100
RUN_ID_PREFIX="hyperstockgat_gpu"

# Arrays of parameters
MARKETS=("nasdaq" "nyse" "sse")
MODELS=("hyperstockgat")
NORMALIZATIONS=("zscore" "minmax" "log1p" "none")

# Loop through all combinations
for market in "${MARKETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for norm in "${NORMALIZATIONS[@]}"; do
            
            # Construct a unique run ID
            run_id="${RUN_ID_PREFIX}_${market}_${model}_${norm}"
            
            echo "Submitting job for ${market} ${model} ${norm}..."
            ./submit_random_search.sh "$market" "$model" "$norm" "$N_TRIALS" "$EPOCHS" "$run_id"
            
        done
    done
done
