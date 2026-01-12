"""
Error Analysis Visualization Script

Generates graphs showing model errors across BERT, DistilBERT, and RoBERTa models.
Outputs: confusion matrices, error distribution by class, per-model error counts, comparative analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Label mapping for FEVER
LABEL_NAMES = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT ENOUGH INFO'}

def load_predictions(model_name, split='validation'):
    """Load predictions CSV for a given model and split."""
    path = os.path.join('outputs', model_name, f'predictions_{split}.csv')
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping {model_name}")
        return None
    df = pd.read_csv(path)
    if 'label' not in df.columns or 'prediction' not in df.columns:
        print(f"Warning: {path} missing required columns")
        return None
    return df

def plot_confusion_matrix(y_true, y_pred, model_name, split='validation', save_dir='outputs/error_analysis'):
    """Plot and save confusion matrix for a model."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[LABEL_NAMES[i] for i in [0, 1, 2]],
                yticklabels=[LABEL_NAMES[i] for i in [0, 1, 2]],
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix ({split})', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix_{split}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {model_name}")

def plot_error_distribution(y_true, y_pred, model_name, split='validation', save_dir='outputs/error_analysis'):
    """Plot error distribution by true class."""
    errors_by_class = {}
    for true_label in [0, 1, 2]:
        mask = y_true == true_label
        errors = y_pred[mask] != y_true[mask]
        errors_by_class[LABEL_NAMES[true_label]] = {
            'total': mask.sum(),
            'errors': errors.sum(),
            'error_rate': errors.sum() / mask.sum() if mask.sum() > 0 else 0
        }
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error counts
    classes = list(errors_by_class.keys())
    error_counts = [errors_by_class[c]['errors'] for c in classes]
    total_counts = [errors_by_class[c]['total'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    ax1.bar(x - width/2, total_counts, width, label='Total Examples', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, error_counts, width, label='Errors', alpha=0.8, color='orangered')
    ax1.set_xlabel('True Class', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f'{model_name} - Error Counts by Class', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Error rates
    error_rates = [errors_by_class[c]['error_rate'] * 100 for c in classes]
    bars = ax2.bar(classes, error_rates, alpha=0.8, color=['steelblue', 'coral', 'mediumseagreen'])
    ax2.set_xlabel('True Class', fontsize=11)
    ax2.set_ylabel('Error Rate (%)', fontsize=11)
    ax2.set_title(f'{model_name} - Error Rate by Class', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(classes, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_error_distribution_{split}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved error distribution: {model_name}")

def plot_comparative_analysis(model_results, split='validation', save_dir='outputs/error_analysis'):
    """Compare errors across all models."""
    if not model_results:
        print("No model results to compare")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    models = list(model_results.keys())
    
    # 1. Overall accuracy comparison
    accuracies = [model_results[m]['accuracy'] * 100 for m in models]
    bars = ax1.bar(models, accuracies, alpha=0.8, color=['steelblue', 'coral', 'mediumseagreen'])
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([min(accuracies) - 5, 100])
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. Macro F1 comparison
    f1_scores = [model_results[m]['macro_f1'] * 100 for m in models]
    bars = ax2.bar(models, f1_scores, alpha=0.8, color=['steelblue', 'coral', 'mediumseagreen'])
    ax2.set_ylabel('Macro F1 (%)', fontsize=11)
    ax2.set_title('Model Macro-F1 Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([min(f1_scores) - 5, 100])
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 3. Total errors per model
    total_errors = [model_results[m]['total_errors'] for m in models]
    bars = ax3.bar(models, total_errors, alpha=0.8, color=['steelblue', 'coral', 'mediumseagreen'])
    ax3.set_ylabel('Number of Errors', fontsize=11)
    ax3.set_title('Total Prediction Errors', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # 4. Error rate by class (grouped bar chart)
    classes = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    x = np.arange(len(classes))
    width = 0.25
    
    for i, model in enumerate(models):
        error_rates = []
        for cls_idx in [0, 1, 2]:
            cls_name = LABEL_NAMES[cls_idx]
            mask = model_results[model]['y_true'] == cls_idx
            if mask.sum() > 0:
                errors = (model_results[model]['y_pred'][mask] != model_results[model]['y_true'][mask]).sum()
                error_rates.append(errors / mask.sum() * 100)
            else:
                error_rates.append(0)
        ax4.bar(x + i*width, error_rates, width, label=model, alpha=0.8)
    
    ax4.set_ylabel('Error Rate (%)', fontsize=11)
    ax4.set_title('Error Rate by Class and Model', fontsize=13, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(classes, rotation=15, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparative_analysis_{split}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved comparative analysis")

def analyze_common_errors(model_results, split='validation', save_dir='outputs/error_analysis', top_n=10):
    """Find examples where all models make mistakes."""
    if len(model_results) < 2:
        print("Need at least 2 models for common error analysis")
        return
    
    models = list(model_results.keys())
    
    # Find indices where all models are wrong
    all_wrong_mask = np.ones(len(model_results[models[0]]['y_true']), dtype=bool)
    for model in models:
        wrong = model_results[model]['y_pred'] != model_results[model]['y_true']
        all_wrong_mask &= wrong
    
    common_errors = all_wrong_mask.sum()
    total = len(all_wrong_mask)
    
    print(f"\n{'='*60}")
    print(f"Common Error Analysis ({split}):")
    print(f"Examples where ALL models failed: {common_errors} / {total} ({common_errors/total*100:.2f}%)")
    print(f"{'='*60}")
    
    # Count by true class
    y_true = model_results[models[0]]['y_true']
    for cls_idx in [0, 1, 2]:
        cls_mask = (y_true == cls_idx) & all_wrong_mask
        cls_total = (y_true == cls_idx).sum()
        print(f"{LABEL_NAMES[cls_idx]}: {cls_mask.sum()} / {cls_total} ({cls_mask.sum()/cls_total*100:.1f}%)")

def main():
    print("="*60)
    print("FEVER Model Error Analysis")
    print("="*60)
    
    save_dir = 'outputs/error_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    models = ['BERT', 'DistilBERT', 'RoBERTa']
    split = 'validation'  # Change to 'test' if you have test labels
    
    model_results = {}
    
    # Process each model
    for model in models:
        df = load_predictions(model, split)
        if df is None:
            continue
        
        y_true = df['label'].values
        y_pred = df['prediction'].values
        
        # Skip if labels are invalid (e.g., all -1)
        if (y_true == -1).all():
            print(f"Skipping {model}: no valid labels")
            continue
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        errors = (y_pred != y_true).sum()
        
        model_results[model] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'accuracy': acc,
            'macro_f1': f1,
            'total_errors': errors
        }
        
        print(f"\n{model}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro-F1: {f1:.4f}")
        print(f"  Total errors: {errors} / {len(y_true)}")
        
        # Generate individual model plots
        plot_confusion_matrix(y_true, y_pred, model, split, save_dir)
        plot_error_distribution(y_true, y_pred, model, split, save_dir)
    
    # Generate comparative plots
    if model_results:
        plot_comparative_analysis(model_results, split, save_dir)
        analyze_common_errors(model_results, split, save_dir)
    
    print(f"\n{'='*60}")
    print(f"All graphs saved to: {save_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
