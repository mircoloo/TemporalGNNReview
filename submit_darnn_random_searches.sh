#!/bin/bash

# Configuration
N_TRIALS=20
EPOCHS=100
RUN_ID_PREFIX="darnn_gpu_retry"

# Arrays of parameters
MARKETS=("nasdaq" "nyse" "sse")
MODELS=("darnn")
NORMALIZATIONS=("zscore" "minmax" "log1p" "none")

# Loop through all combinations
for market in "${MARKETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for norm in "${NORMALIZATIONS[@]}"; do
            
            # Construct a unique run ID
            run_id="${RUN_ID_PREFIX}_${market}_${model}_${norm}"
            
            echo "Submitting job for: Market=${market}, Model=${model}, Norm=${norm}"
            
            # Call the submission script
            ./submit_random_search.sh "$market" "$model" "$norm" "$N_TRIALS" "$EPOCHS" "$run_id"
            
            # Optional: sleep briefly to avoid overwhelming the scheduler if needed
            sleep 0.5
        done
    done
done

echo "All DARNN jobs submitted!"
