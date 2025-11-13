#!/usr/bin/env python3
"""
Script to synchronize all analyses across NASDAQ, NYSE, and SSE thesis analysis notebooks.
This ensures all three market notebooks have identical analysis structure.
"""

import json
from pathlib import Path

# Define the analyses to be added after specific sections
ANALYSES_TO_ADD = {
    "feature_entropy": {
        "after_section": "1.5",  # After feature correlation
        "markdown": """### 1.6. Feature Entropy Analysis

Feature entropy measures the predictability of a feature. Higher entropy implies less predictability and more information content.""",
        "code": """# Calculate entropy for each feature
def calculate_entropy(series, bins=50):
    \"\"\"Calculates the entropy of a continuous series.\"\"\"
    counts = np.histogram(series, bins=bins)[0]
    return entropy(counts, base=2)

feature_entropy = {feat: calculate_entropy(features_df[feat]) for feat in train_analyzer.features}
entropy_series = pd.Series(feature_entropy).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=entropy_series.index, y=entropy_series.values)
plt.title(f'Feature Entropy ({MARKET})')
plt.ylabel('Entropy (bits)')
plt.xticks(rotation=45)
save_figure('feature_entropy')
plt.show()

print("--- Feature Entropy ---")
display(entropy_series.to_frame('Entropy'))

# Save LaTeX table
print("\\n--- LaTeX Table: Feature Entropy ---")
print(entropy_series.to_frame('Entropy').to_latex(caption=f'Feature entropy for {MARKET} dataset.',
                                                   label=f'tab:{MARKET.lower()}_feature_entropy',
                                                   float_format="%.3f"))"""
    },
    "adjacency_visualization": {
        "markdown": """### 2.1. Adjacency Matrix Visualization and Sparsity

We visualize the adjacency matrix of a selected snapshot and compute sparsity statistics to understand the graph density.""",
        "code": """# Visualize a selected adjacency matrix
snapshot_index = 10  # Select a representative snapshot
plt.figure(figsize=(10, 8))
sns.heatmap(train_adjs[snapshot_index], cmap='viridis', cbar_kws={'label': 'Edge Weight'})
plt.title(f'Adjacency Matrix Heatmap ({MARKET}, Snapshot {snapshot_index})')
plt.xlabel('Node Index')
plt.ylabel('Node Index')
save_figure(f'adjacency_matrix_snapshot_{snapshot_index}')
plt.show()

# Sparsity and Edge Weight Analysis
sparsity_data = []
for name, adjs in adjacency_matrices.items():
    if adjs is None or len(adjs) == 0:
        continue
    sparsity = 1.0 - np.mean([np.count_nonzero(adj) / adj.size for adj in adjs])
    sparsity_data.append({
        'Dataset': name,
        'Sparsity': sparsity,
        'Connectivity': analyzers[name].get_connectivity()
    })

sparsity_df = pd.DataFrame(sparsity_data).set_index('Dataset')
print("--- Graph Sparsity and Connectivity ---")
display(sparsity_df)

# Save LaTeX table
print("\\n--- LaTeX Table: Sparsity ---")
print(sparsity_df.to_latex(caption=f'Graph sparsity and connectivity for {MARKET} datasets.',
                           label=f'tab:{MARKET.lower()}_sparsity',
                           float_format="%.4f"))"""
    }
}

def main():
    print("This script outlines the structure for syncing analyses.")
    print("The actual implementation would use the edit_notebook_file tool.")
    print("Please use the Copilot interface to execute the cell insertions.")

if __name__ == "__main__":
    main()
