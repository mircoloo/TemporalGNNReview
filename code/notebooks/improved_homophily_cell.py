# ===================================================================
# IMPROVED HOMOPHILY VISUALIZATION - ALL SPLITS
# This cell replaces the old homophily_over_time visualization
# ===================================================================

# Compute homophily scores for all splits
homophily_data = {}
for name, analyzer in analyzers.items():
    homophily_scores = [analyzer.get_homophily_score(i) for i in range(analyzer.num_snapshots)]
    homophily_data[name] = {
        'scores': homophily_scores,
        'mean': np.mean(homophily_scores),
        'std': np.std(homophily_scores),
        'min': np.min(homophily_scores),
        'max': np.max(homophily_scores)
    }

# Create comprehensive visualization with 2 rows x 2 columns
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ===================================================================
# PLOT 1: Time Series for All Splits (Top Left)
# ===================================================================
ax1 = fig.add_subplot(gs[0, :])  # Span full width of first row

colors = {'Train': '#3498db', 'Validation': '#e67e22', 'Test': '#2ecc71'}
for name, data in homophily_data.items():
    ax1.plot(data['scores'], marker='.', linestyle='-', label=name, 
             color=colors[name], linewidth=2, markersize=8, alpha=0.7)
    # Add mean line
    ax1.axhline(y=data['mean'], color=colors[name], linestyle='--', 
               linewidth=1.5, alpha=0.5)

ax1.set_title(f'{MARKET} - Homophily Score Over Time (All Splits)', 
             fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Snapshot Index', fontsize=12, fontweight='bold')
ax1.set_ylabel('Homophily Score', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11, loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# ===================================================================
# PLOT 2: Distribution Comparison (Bottom Left)
# ===================================================================
ax2 = fig.add_subplot(gs[1, 0])

# Prepare data for histogram
all_scores = []
all_labels = []
for name, data in homophily_data.items():
    all_scores.extend(data['scores'])
    all_labels.extend([name] * len(data['scores']))

scores_df = pd.DataFrame({'Homophily': all_scores, 'Split': all_labels})

# Plot overlapping histograms
for name, color in colors.items():
    subset = scores_df[scores_df['Split'] == name]['Homophily']
    ax2.hist(subset, bins=20, alpha=0.5, label=name, color=color, edgecolor='black')
    # Add mean line
    ax2.axvline(x=homophily_data[name]['mean'], color=color, linestyle='--', 
               linewidth=2, label=f'{name} Mean')

ax2.set_title('Homophily Distribution Comparison', fontsize=13, fontweight='bold')
ax2.set_xlabel('Homophily Score', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9, loc='best')
ax2.grid(True, axis='y', alpha=0.3)

# ===================================================================
# PLOT 3: Box Plot Comparison (Bottom Right)
# ===================================================================
ax3 = fig.add_subplot(gs[1, 1])

# Create box plot
box_data = [homophily_data[name]['scores'] for name in ['Train', 'Validation', 'Test']]
bp = ax3.boxplot(box_data, labels=['Train', 'Validation', 'Test'],
                 patch_artist=True, widths=0.6,
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))

# Color boxes
for patch, name in zip(bp['boxes'], ['Train', 'Validation', 'Test']):
    patch.set_facecolor(colors[name])
    patch.set_alpha(0.6)

# Add mean markers
means = [homophily_data[name]['mean'] for name in ['Train', 'Validation', 'Test']]
ax3.plot(range(1, 4), means, 'D', color='darkred', markersize=10, 
        label='Mean', zorder=3)

ax3.set_title('Homophily Score Distribution', fontsize=13, fontweight='bold')
ax3.set_ylabel('Homophily Score', fontsize=11, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, axis='y', alpha=0.3)

plt.suptitle(f'{MARKET} - Comprehensive Homophily Analysis', 
            fontsize=16, fontweight='bold', y=0.98)

save_figure('homophily_comprehensive_all_splits')
plt.show()

# ===================================================================
# PRINT STATISTICS TABLE
# ===================================================================
print("\n" + "="*80)
print(f"{MARKET} - HOMOPHILY STATISTICS SUMMARY")
print("="*80)

stats_table = []
for name in ['Train', 'Validation', 'Test']:
    data = homophily_data[name]
    stats_table.append([
        name,
        f"{data['mean']:.4f}",
        f"{data['std']:.4f}",
        f"{data['min']:.4f}",
        f"{data['max']:.4f}",
        f"{len(data['scores'])}"
    ])

from tabulate import tabulate
headers = ['Split', 'Mean', 'Std Dev', 'Min', 'Max', '# Snapshots']
print(tabulate(stats_table, headers=headers, tablefmt='pretty'))
print("="*80 + "\n")

# ===================================================================
# LATEX TABLE GENERATION
# ===================================================================
print("--- LaTeX Table: Homophily Statistics ---\n")

latex_table = r"\begin{table}[h]" + "\n"
latex_table += r"    \centering" + "\n"
latex_table += r"    \begin{tabular}{lcccc}" + "\n"
latex_table += r"        \toprule" + "\n"
latex_table += r"        \textbf{Split} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\" + "\n"
latex_table += r"        \midrule" + "\n"

for name in ['Train', 'Validation', 'Test']:
    data = homophily_data[name]
    latex_table += f"        {name:12s} & {data['mean']:.4f} & {data['std']:.4f} & {data['min']:.4f} & {data['max']:.4f} \\\\\n"

latex_table += r"        \bottomrule" + "\n"
latex_table += r"    \end{tabular}" + "\n"
latex_table += f"    \\caption{{Homophily score statistics for {MARKET} dataset across train, validation, and test splits.}}\n"
latex_table += f"    \\label{{tab:{MARKET.lower()}_homophily_stats}}\n"
latex_table += r"\end{table}"

print(latex_table)

# Save LaTeX table to file
latex_file_path = FIGURE_OUTPUT_PATH / f"{MARKET}_homophily_stats.tex"
with open(latex_file_path, 'w') as f:
    f.write(latex_table)
print(f"\n✓ LaTeX table saved to: {latex_file_path}")
