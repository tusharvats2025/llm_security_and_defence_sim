"""
Visualization script for benchmark results.
Creates comprehensive charts and reports.
Enhanced with better error handling and additional metrics.
"""
import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
import argparse

# add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_dir: str = "results"):
    """Load benchmark results from files."""
    result_path = Path(results_dir)

    # Check if files exist
    csv_path = result_path / "benchmark_results.csv"
    json_path = result_path / "metrics_summary.json"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Load CSV data
    df = pd.read_csv(csv_path)

    # Load metrics
    with open(json_path, 'r') as f:
        metrics = json.load(f)

    return df, metrics


def create_overview_chart(metrics, output_path: str = "results/overview_chart.png"):
    """Create overview bar chart of main metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Attack Success\n(Vulnerable)', 'Defense Success\n(Protected)',
                  'False Positive\n(Benign Blocked)', 'False Negative\n(Attacks Missed)']
    
    values = [
        metrics['overall_metrics'].get('attack_success_rate', 0),
        metrics['overall_metrics'].get('defense_success_rate', 0),
        metrics['overall_metrics'].get('false_positive_rate', 0),
        metrics['overall_metrics'].get('false_negative_rate', 0)
    ]
    colors = ['#e74c3c', '#27ae60', '#f39c12', '#e67e22']

    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')

    # add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('LLM Security Framework - Performance Overview', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved overview chart: {output_path}")
    plt.close()


def create_category_comparison(metrics, output_path: str = "results/category_breakdown.png"):
    """Create comparison chart by attack category."""
    category_data = metrics['overall_metrics'].get('category_breakdown', {})

    if not category_data:
        print("⚠️ No category breakdown data available.")
        return

    categories = list(category_data.keys())
    attack_rates = [category_data[cat].get('attack_success_rate', 0) for cat in categories]
    defense_rates = [category_data[cat].get('defense_success_rate', 0) for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, attack_rates, width, label='Attack Success (Vulnerable)',
                   color='#e74c3c', alpha=0.7)
    bars2 = ax.bar(x + width/2, defense_rates, width, label='Defense Success (Protected)',
                   color='#27ae60', alpha=0.7)
    
    # Add values labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.0f}%',
                    ha='center', va='bottom', fontsize=9)
            
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Performance by Attack Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories],
                       rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved category breakdown: {output_path}")
    plt.close()


def create_confusion_matrix_heatmap(metrics, output_path: str = "results/confusion_matrix.png"):
    """Create confusion matrix heatmap."""
    cm_data = metrics.get('confusion_matrix', {})

    if not cm_data:
        print("⚠️ No confusion matrix data available.")
        return

    matrix = np.array([
        [cm_data.get('true_positives', 0), cm_data.get('false_negatives', 0)],
        [cm_data.get('false_positives', 0), cm_data.get('true_negatives', 0)]
    ])

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(matrix, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=['Malicious', 'Benign'],
                yticklabels=['Blocked', 'Passed'],
                cbar_kws={'label': 'Count'},
                ax=ax, linewidths=2, linecolor='black')
    ax.set_title('Defense Performance - Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual Category', fontsize=12)
    ax.set_ylabel('Defense Action', fontsize=12)

    # Add metrics text with F1 score
    precision = cm_data.get('precision', 0)
    recall = cm_data.get('recall', 0)
    f1_score = cm_data.get('f1_score', 0)
    accuracy = cm_data.get('accuracy', 0)
    
    textstr = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1_score:.3f}\nAccuracy: {accuracy:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix heatmap: {output_path}")
    plt.close()


def create_execution_time_chart(df, output_path: str = "results/execution_times.png"):
    """Create execution time analysis chart."""
    if 'execution_time' not in df.columns:
        print("⚠️ Execution time data not available in results.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(df['execution_time'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(df['execution_time'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df["execution_time"].mean():.3f}s')
    ax1.set_xlabel('Execution Time (seconds)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Execution Times', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Boxplot by category
    categories = df['category'].unique()
    data_by_category = [df[df['category'] == cat]['execution_time'].values for cat in categories]

    bp = ax2.boxplot(data_by_category, 
                     labels=[cat.replace('_', ' ').title() for cat in categories], 
                     patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)

    ax2.set_ylabel('Execution Time (seconds)', fontsize=11)
    ax2.set_title('Execution Time by Category', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)  # FIXED: was trick_params
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved execution time analysis: {output_path}")
    plt.close()


def create_comprehensive_report(df, metrics, output_path: str = "results/comprehensive_report.png"):
    """Create a comprehensive multi-panel report."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Main metrics
    ax1 = fig.add_subplot(gs[0, :])
    categories = ['Attack Success', 'Defense Success', 'False Positives', 'False Negatives']
    values = [
        metrics['overall_metrics'].get('attack_success_rate', 0),
        metrics['overall_metrics'].get('defense_success_rate', 0),
        metrics['overall_metrics'].get('false_positive_rate', 0),
        metrics['overall_metrics'].get('false_negative_rate', 0)
    ]
    colors = ['#e74c3c', '#27ae60', '#f39c12', '#e67e22']
    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Overall Performance Metrics', fontweight='bold', fontsize=14)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Category breakdown
    ax2 = fig.add_subplot(gs[1, :2])
    category_data = metrics['overall_metrics'].get('category_breakdown', {})
    
    if category_data:
        cats = list(category_data.keys())
        attack_rates = [category_data[c].get('attack_success_rate', 0) for c in cats]
        defense_rates = [category_data[c].get('defense_success_rate', 0) for c in cats]

        x = np.arange(len(cats))
        width = 0.35
        ax2.bar(x - width/2, attack_rates, width, label='Vulnerable', color='#e74c3c', alpha=0.7)
        ax2.bar(x + width/2, defense_rates, width, label='Protected', color='#27ae60', alpha=0.7)
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Performance by Category', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([c.replace('_', ' ')[:15] for c in cats], rotation=45, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No category data available', ha='center', va='center')
        ax2.set_title('Performance by Category', fontweight='bold')

    # 3. Confusion matrix
    ax3 = fig.add_subplot(gs[1, 2])
    cm_data = metrics.get('confusion_matrix', {})
    matrix = np.array([
        [cm_data.get('true_positives', 0), cm_data.get('false_negatives', 0)],
        [cm_data.get('false_positives', 0), cm_data.get('true_negatives', 0)]
    ])
    im = ax3.imshow(matrix, cmap='RdYlGn', aspect='auto')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Malicious', 'Benign'])
    ax3.set_yticklabels(['Blocked', 'Passed'])
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, matrix[i, j], ha='center', va='center', 
                    color='white' if matrix[i, j] > matrix.max()/2 else 'black',
                    fontweight='bold', fontsize=14)
    ax3.set_title('Confusion Matrix', fontweight='bold')

    # 4. Test distribution
    ax4 = fig.add_subplot(gs[2, 0])
    if 'category' in df.columns:
        category_counts = df['category'].value_counts()
        ax4.pie(category_counts.values, labels=[c.replace('_', ' ')[:20] for c in category_counts.index],
                autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
        ax4.set_title('Test Distribution', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No category data', ha='center', va='center')
        ax4.set_title('Test Distribution', fontweight='bold')

    # 5. Success/Failure rates
    ax5 = fig.add_subplot(gs[2, 1])
    attack_success_count = df['attack_success'].sum() if 'attack_success' in df.columns else 0
    defense_success_count = df['defense_success'].sum() if 'defense_success' in df.columns else 0

    data = [
        ['Attacks Succeeded', attack_success_count, '#e74c3c'],
        ['Defenses Succeeded', defense_success_count, '#27ae60'],
    ]

    y_pos = np.arange(len(data))
    counts = [d[1] for d in data]
    colors_bar = [d[2] for d in data]

    ax5.barh(y_pos, counts, color=colors_bar, alpha=0.7)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([d[0] for d in data])
    ax5.set_xlabel('Count')
    ax5.set_title('Attack vs Defense Outcomes', fontweight='bold')

    for i, v in enumerate(counts):
        ax5.text(v, i, f' {v}', va='center', fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)

    # 6. Metrics summary
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    f1_score = cm_data.get('f1_score', 0)
    accuracy = cm_data.get('accuracy', 0)
    
    summary_text = f"""
METRICS SUMMARY
Total Tests: {metrics['overall_metrics'].get('total_tests', 0)}
Precision: {cm_data.get('precision', 0):.3f}
Recall: {cm_data.get('recall', 0):.3f}
F1 Score: {f1_score:.3f}
Accuracy: {accuracy:.3f}

TP: {cm_data.get('true_positives', 0)}
TN: {cm_data.get('true_negatives', 0)}
FP: {cm_data.get('false_positives', 0)}
FN: {cm_data.get('false_negatives', 0)}
"""
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, 
              fontsize=11, verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
              family='monospace')
    
    plt.suptitle('LLM Security Framework - Comprehensive Report',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive report: {output_path}")
    plt.close()


def generate_all_visualizations(results_dir: str = "results"):
    """Generate all visualization charts."""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60 + "\n")

    try:
        df, metrics = load_results(results_dir)
        print(f"✓ Loaded {len(df)} test results\n")

        # Generate all charts (no trailing backslash)
        create_overview_chart(metrics, f"{results_dir}/overview_chart.png")
        create_category_comparison(metrics, f"{results_dir}/category_breakdown.png")
        create_confusion_matrix_heatmap(metrics, f"{results_dir}/confusion_matrix.png")
        create_execution_time_chart(df, f"{results_dir}/execution_times.png")
        create_comprehensive_report(df, metrics, f"{results_dir}/comprehensive_report.png")
        
        print("\n" + "=" * 60)
        print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nView your results in the '{results_dir}' directory")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Run 'python demos/run_benchmarks.py' first to generate results")
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualization charts")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing benchmark results"
    )

    args = parser.parse_args()
    generate_all_visualizations(args.results_dir)
    



