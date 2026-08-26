"""Result export: Excel tables, confusion matrices, comparison charts.

The 'Method' column holds the method label produced by ``methods`` —
``Baseline`` or ``CWA (K=n)``.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MACRO_COLUMNS = ['Model', 'Method', 'Test Loss', 'Accuracy (%)', 'Precision (%)',
                 'Recall (%)', 'F1-Score (%)', 'AUC (%)']

PER_CLASS_COLUMNS = ['Model', 'Method', 'Class', 'Accuracy (%)', 'Precision (%)',
                     'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'AUC (%)',
                     'Support']

CHART_METRICS = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'AUC (%)']


def results_to_frames(all_model_results):
    """Flatten nested results into (macro, per-class) dataframes."""
    macro_rows = []
    per_class_rows = []

    for model_name, method_results in all_model_results.items():
        for method_name, result in method_results.items():
            macro_rows.append({
                'Model': model_name,
                'Method': method_name,
                **result['metrics']
            })

            for cls_name, cls_metrics in result['per_class'].items():
                per_class_rows.append({
                    'Model': model_name,
                    'Method': method_name,
                    'Class': cls_name,
                    **cls_metrics
                })

    df = pd.DataFrame(macro_rows)[MACRO_COLUMNS]
    df_pc = pd.DataFrame(per_class_rows)[PER_CLASS_COLUMNS]
    return df, df_pc


def export_results_to_excel(all_model_results, output_path):
    """
    Export all results to Excel with separate sheets for macro and per-class metrics

    Args:
        all_model_results: {model_name: {method_name: result}}
        output_path: Path to save Excel file

    Returns:
        df: Macro-averaged results dataframe
    """
    df, df_pc = results_to_frames(all_model_results)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Overall Metrics', index=False)
        df_pc.to_excel(writer, sheet_name='Per-Class Metrics', index=False)

    print(f"\n✓ Results exported to: {output_path}")
    print(f"  - Sheet 'Overall Metrics': Macro-averaged metrics")
    print(f"  - Sheet 'Per-Class Metrics': Per-class breakdown")

    return df


def save_confusion_matrices(all_model_results, output_dir, class_names=None):
    """Save a confusion-matrix heatmap per model and method."""
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)

    for model_name, method_results in all_model_results.items():
        for method_name, result in method_results.items():
            cm = result['confusion_matrix']

            fig, ax = plt.subplots(figsize=(max(8, len(cm) * 1.2), max(6, len(cm) * 1.0)))

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names if class_names else range(len(cm)),
                        yticklabels=class_names if class_names else range(len(cm)),
                        ax=ax)

            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title(f'{model_name} - {method_name}\nConfusion Matrix',
                         fontsize=13, fontweight='bold')

            plt.tight_layout()

            safe_method = (method_name.replace(' ', '_').replace('(', '')
                           .replace(')', '').replace('=', ''))
            filepath = os.path.join(cm_dir, f'{model_name}_{safe_method}_cm.png')
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            plt.close()

    print(f"✓ Confusion matrices saved to: {cm_dir}")


def create_performance_charts(df, output_dir):
    """Create one comprehensive Baseline-vs-CWA comparison chart."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Model Performance Comparison Across All Methods',
                 fontsize=16, fontweight='bold')

    models = df['Model'].unique()
    methods = df['Method'].unique()

    for idx, metric in enumerate(CHART_METRICS):
        ax = axes[idx // 3, idx % 3]

        x = np.arange(len(models))
        width = 0.15

        for i, method in enumerate(methods):
            method_data = df[df['Method'] == method]
            values = [method_data[method_data['Model'] == model][metric].values[0]
                      for model in models]
            ax.bar(x + i * width, values, width, label=method)

        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    # Summary table in the last subplot
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = "Best Models per Method:\n\n"
    for method in methods:
        method_df = df[df['Method'] == method]
        best_idx = method_df['F1-Score (%)'].idxmax()
        summary_text += (f"{method}:\n  {method_df.loc[best_idx, 'Model']} "
                         f"(F1: {method_df.loc[best_idx, 'F1-Score (%)']:.2f}%)\n\n")

    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Chart saved to: {chart_path}")
