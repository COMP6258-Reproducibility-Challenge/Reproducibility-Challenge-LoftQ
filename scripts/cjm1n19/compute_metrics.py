import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics_from_csv(
    csv_matched_path, 
    csv_mismatched_path, 
    output_dir="./metrics_from_csv",
    label_map=None
):
    """
    Calculate evaluation metrics from prediction CSVs for MNLI.
    
    Args:
        csv_matched_path: Path to matched predictions CSV
        csv_mismatched_path: Path to mismatched predictions CSV
        output_dir: Directory to save the evaluation results
        label_map: Optional mapping from string labels to numeric IDs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prediction CSVs
    logger.info(f"Loading predictions from {csv_matched_path} and {csv_mismatched_path}")
    matched_df = pd.read_csv(csv_matched_path)
    mismatched_df = pd.read_csv(csv_mismatched_path)
    
    # Load MNLI validation datasets to get true labels
    logger.info("Loading MNLI validation datasets for ground truth")
    dataset = load_dataset("glue", "mnli")
    val_matched = dataset["validation_matched"]
    val_mismatched = dataset["validation_mismatched"]
    
    # Convert to DataFrames
    val_matched_df = pd.DataFrame({
        "idx": val_matched["idx"],
        "true_label": val_matched["label"]
    })
    
    print(val_matched_df[:10]["label"])
    sys.exit()
    
    val_mismatched_df = pd.DataFrame({
        "idx": val_mismatched["idx"],
        "true_label": val_mismatched["label"]
    })
    
    # Merge predictions with true labels
    matched_merged = pd.merge(matched_df, val_matched_df, on="idx")
    mismatched_merged = pd.merge(mismatched_df, val_mismatched_df, on="idx")
    
    # Define label mapping if not provided
    if label_map is None:
        label_map = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }
    
    # Convert string labels to numeric if necessary
    if isinstance(matched_merged["prediction"].iloc[0], str):
        matched_merged["prediction_id"] = matched_merged["prediction"].map(label_map)
    else:
        matched_merged["prediction_id"] = matched_merged["prediction"]
    
    if isinstance(mismatched_merged["prediction"].iloc[0], str):
        mismatched_merged["prediction_id"] = mismatched_merged["prediction"].map(label_map)
    else:
        mismatched_merged["prediction_id"] = mismatched_merged["prediction"]
    
    # Define function to compute all metrics
    def compute_all_metrics(df):
        # Get predictions and true labels
        y_true = df["true_label"].values
        y_pred = df["prediction_id"].values
        
        # Calculate accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # Calculate precision, recall, F1 per class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2]
        )
        
        # Calculate weighted and macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Combine all metrics
        metrics = {
            "accuracy": float(acc),
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "class_metrics": {
                "entailment": {
                    "precision": float(precision[0]),
                    "recall": float(recall[0]),
                    "f1": float(f1[0]),
                    "support": int(support[0])
                },
                "neutral": {
                    "precision": float(precision[1]),
                    "recall": float(recall[1]),
                    "f1": float(f1[1]),
                    "support": int(support[1])
                },
                "contradiction": {
                    "precision": float(precision[2]),
                    "recall": float(recall[2]),
                    "f1": float(f1[2]),
                    "support": int(support[2])
                }
            }
        }
        
        return metrics, y_pred
    
    # Compute metrics for matched validation set
    logger.info("Computing metrics for matched validation set")
    matched_metrics, matched_preds = compute_all_metrics(matched_merged)
    
    # Compute metrics for mismatched validation set
    logger.info("Computing metrics for mismatched validation set")
    mismatched_metrics, mismatched_preds = compute_all_metrics(mismatched_merged)
    
    # Create confusion matrices
    matched_cm = confusion_matrix(matched_merged["true_label"], matched_preds)
    mismatched_cm = confusion_matrix(mismatched_merged["true_label"], mismatched_preds)
    
    # Plot and save confusion matrices
    labels = ["entailment", "neutral", "contradiction"]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matched_cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Matched Validation Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'matched_confusion_matrix.png'))
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(mismatched_cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Mismatched Validation Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mismatched_confusion_matrix.png'))
    plt.close()
    
    # Generate detailed classification reports
    matched_report = classification_report(
        matched_merged["true_label"], matched_preds, 
        target_names=labels, output_dict=True
    )
    
    mismatched_report = classification_report(
        mismatched_merged["true_label"], mismatched_preds, 
        target_names=labels, output_dict=True
    )
    
    # Save metrics to JSON files
    with open(os.path.join(output_dir, 'matched_metrics.json'), 'w') as f:
        json.dump(matched_metrics, f, indent=4)
    
    with open(os.path.join(output_dir, 'mismatched_metrics.json'), 'w') as f:
        json.dump(mismatched_metrics, f, indent=4)
    
    with open(os.path.join(output_dir, 'matched_classification_report.json'), 'w') as f:
        json.dump(matched_report, f, indent=4)
    
    with open(os.path.join(output_dir, 'mismatched_classification_report.json'), 'w') as f:
        json.dump(mismatched_report, f, indent=4)
    
    # Create a summary table
    summary = {
        "Metric": ["Accuracy", "F1 Weighted", "F1 Macro", "Precision Weighted", "Recall Weighted"],
        "Matched": [
            matched_metrics["accuracy"], 
            matched_metrics["f1_weighted"],
            matched_metrics["f1_macro"],
            matched_metrics["precision_weighted"],
            matched_metrics["recall_weighted"]
        ],
        "Mismatched": [
            mismatched_metrics["accuracy"], 
            mismatched_metrics["f1_weighted"],
            mismatched_metrics["f1_macro"],
            mismatched_metrics["precision_weighted"],
            mismatched_metrics["recall_weighted"]
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    
    # Print summary metrics
    logger.info("\n=== EVALUATION SUMMARY ===")
    logger.info(f"Matched validation accuracy: {matched_metrics['accuracy']:.4f}")
    logger.info(f"Matched validation F1 weighted: {matched_metrics['f1_weighted']:.4f}")
    logger.info(f"Mismatched validation accuracy: {mismatched_metrics['accuracy']:.4f}")
    logger.info(f"Mismatched validation F1 weighted: {mismatched_metrics['f1_weighted']:.4f}")
    
    logger.info(f"All metrics saved to {output_dir}")
    
    return {
        "matched": matched_metrics,
        "mismatched": mismatched_metrics
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate metrics from prediction CSVs")
    parser.add_argument("--csv_matched", type=str, required=True, help="Path to matched predictions CSV")
    parser.add_argument("--csv_mismatched", type=str, required=True, help="Path to mismatched predictions CSV")
    parser.add_argument("--output_dir", type=str, default="./metrics_from_csv", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Calculate metrics from CSVs
    calculate_metrics_from_csv(
        csv_matched_path=args.csv_matched,
        csv_mismatched_path=args.csv_mismatched,
        output_dir=args.output_dir
    )