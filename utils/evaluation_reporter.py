# utils/evaluation_reporter.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
import editdistance
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import unicodedata # For character analysis later if needed
import logging
import json
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def calculate_corpus_metrics(predictions, ground_truths):
    """Calculates corpus-level CER and WER."""
    total_edit_distance_char = 0
    total_chars = 0
    total_edit_distance_word = 0
    total_words = 0

    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        logger.warning("Invalid input for metric calculation. Returning high error rates.")
        return 1.0, 1.0 # Return worst case

    for pred_str, gt_str in zip(predictions, ground_truths):
        try:
            total_edit_distance_char += editdistance.eval(pred_str, gt_str)
            total_chars += len(gt_str) # Normalize by ground truth length
        except Exception as cer_e:
            logger.warning(f"Could not calculate char edit distance for GT='{gt_str}', Pred='{pred_str}': {cer_e}")

        try:
            pred_words = pred_str.split()
            gt_words = gt_str.split()
            total_edit_distance_word += editdistance.eval(pred_words, gt_words)
            total_words += len(gt_words) # Normalize by ground truth length
        except Exception as wer_e:
            logger.warning(f"Could not calculate word edit distance for GT='{gt_str}', Pred='{pred_str}': {wer_e}")

    cer = total_edit_distance_char / total_chars if total_chars > 0 else 1.0
    wer = total_edit_distance_word / total_words if total_words > 0 else 1.0
    return cer, wer

def analyze_errors(df_results):
    """Analyzes common error types and patterns."""
    error_df = df_results[df_results['Prediction'] != df_results['GroundTruth']]
    analysis = {
        'total_errors': len(error_df),
        'error_rate': len(error_df) / len(df_results) if len(df_results) > 0 else 0,
        'common_char_errors': {},
        'common_word_errors': {},
        'length_mismatches': 0,
        'examples': {
            'correct': [],
            'incorrect': [],
            'longest_correct': [],
            'longest_incorrect': [],
        }
    }

    if analysis['total_errors'] == 0:
        logger.info("No errors found in predictions.")
        return analysis # Return early if no errors

    char_errors = Counter()
    word_errors = Counter()

    for _, row in error_df.iterrows():
        pred = row['Prediction']
        gt = row['GroundTruth']

        # Length mismatch
        if len(pred) != len(gt):
            analysis['length_mismatches'] += 1

        # Character-level differences (simple alignment)
        for i in range(min(len(pred), len(gt))):
            if pred[i] != gt[i]:
                char_errors[f"{gt[i]} -> {pred[i]}"] += 1

        # Word-level differences
        pred_words = pred.split()
        gt_words = gt.split()
        # This is a simple check, editdistance provides more detail
        if pred_words != gt_words:
             # Log common word substitutions/insertions/deletions? More complex.
             # For now, just count as a word error instance.
             pass # Add more detailed word error analysis if needed

    analysis['common_char_errors'] = dict(char_errors.most_common(20))
    # analysis['common_word_errors'] = ... # Implement if needed

    # Find examples
    correct_samples = df_results[df_results['Prediction'] == df_results['GroundTruth']].head(5)
    incorrect_samples = error_df.head(10) # Show more incorrect examples
    df_results['gt_length'] = df_results['GroundTruth'].apply(len)
    longest_correct = df_results[df_results['Prediction'] == df_results['GroundTruth']].nlargest(5, 'gt_length')
    longest_incorrect = error_df.nlargest(5, 'gt_length')

    analysis['examples']['correct'] = correct_samples.to_dict('records')
    analysis['examples']['incorrect'] = incorrect_samples.to_dict('records')
    analysis['examples']['longest_correct'] = longest_correct.to_dict('records')
    analysis['examples']['longest_incorrect'] = longest_incorrect.to_dict('records')


    return analysis


def generate_visualizations(df_results, output_dir):
    """Generates plots related to performance."""
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    logger.info(f"Saving visualizations to: {viz_dir}")

    # 1. Length vs CER (if CER data is available per sample - needs modification)
    # Placeholder: Distribution of prediction lengths vs GT lengths
    try:
        df_results['pred_length'] = df_results['Prediction'].apply(len)
        df_results['gt_length'] = df_results['GroundTruth'].apply(len)

        plt.figure(figsize=(12, 6))
        sns.histplot(df_results['gt_length'], color='blue', label='Ground Truth Length', kde=True, stat="density", linewidth=0, alpha=0.5)
        sns.histplot(df_results['pred_length'], color='red', label='Prediction Length', kde=True, stat="density", linewidth=0, alpha=0.5)
        plt.title('Distribution of Ground Truth vs. Prediction Lengths')
        plt.xlabel('Length (Characters)')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'length_distribution.png'))
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate length distribution plot: {e}")

    # Add more plots: e.g., confidence distribution if available, confusion matrix for chars

    return viz_dir

def create_html_report(report_path, summary_stats, error_analysis, viz_dir):
    """Creates an HTML report summarizing the evaluation results."""
    logger.info(f"Generating HTML report at: {report_path}")

    # Basic HTML structure
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CTC OCR Evaluation Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary-box {{ border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; background-color: #f9f9f9; }}
        .metrics td {{ font-weight: bold; color: #0056b3; }}
        .error {{ color: red; }}
        .correct {{ color: green; }}
        img {{ max-width: 800px; height: auto; display: block; margin: 10px auto; border: 1px solid #ccc; }}
        pre {{ background-color: #eee; padding: 10px; border: 1px solid #ccc; white-space: pre-wrap; word-wrap: break-word; }}
    </style>
</head>
<body>
    <h1>CTC OCR Model Evaluation Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Model Path: {summary_stats.get('model_path', 'N/A')}</p>
    <p>Test Dataset: {summary_stats.get('dataset_name', 'N/A')}</p>

    <h2>Summary Metrics</h2>
    <div class="summary-box metrics">
        <table>
            <tr><td>Total Samples</td><td>{summary_stats.get('total_samples', 'N/A')}</td></tr>
            <tr><td>Validation Loss (Avg)</td><td>{summary_stats.get('avg_loss', 'N/A'):.4f}</td></tr>
            <tr><td>Character Error Rate (CER)</td><td>{summary_stats.get('cer', 'N/A'):.4f}</td></tr>
            <tr><td>Word Error Rate (WER)</td><td>{summary_stats.get('wer', 'N/A'):.4f}</td></tr>
        </table>
    </div>

    <h2>Error Analysis</h2>
    <div class="summary-box">
        <p>Total Errors: {error_analysis.get('total_errors', 0)} ({error_analysis.get('error_rate', 0):.2%})</p>
        <p>Length Mismatches: {error_analysis.get('length_mismatches', 0)}</p>
        <h3>Most Common Character Errors (GT -> Pred)</h3>
        {f'<pre>{json.dumps(error_analysis.get("common_char_errors", {}), indent=2)}</pre>' if error_analysis.get("common_char_errors") else '<p>No character errors recorded.</p>'}
    </div>

    <h2>Visualizations</h2>
    """
    # Add images from viz_dir
    if viz_dir and os.path.isdir(viz_dir):
        try:
            for filename in sorted(os.listdir(viz_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Use relative path for embedding in HTML
                    relative_path = os.path.join("visualizations", filename)
                    html += f'<h3>{os.path.splitext(filename)[0].replace("_", " ").title()}</h3>\n'
                    html += f'<img src="{relative_path}" alt="{filename}">\n'
        except Exception as e:
            html += f"<p>Error loading visualizations: {e}</p>"
    else:
        html += "<p>No visualizations generated.</p>"

    html += """
    <h2>Examples</h2>
    """
    # Add examples tables
    for category, examples in error_analysis['examples'].items():
        if examples:
            html += f"<h3>{category.replace('_', ' ').title()}</h3>\n<table>\n"
            html += "<tr><th>#</th><th>Ground Truth</th><th>Prediction</th><th>Correct?</th><th>Length (GT)</th></tr>\n"
            for i, ex in enumerate(examples):
                is_correct = ex['GroundTruth'] == ex['Prediction']
                correct_class = "correct" if is_correct else "error"
                html += f"<tr><td>{i+1}</td><td>{ex['GroundTruth']}</td><td>{ex['Prediction']}</td><td class='{correct_class}'>{is_correct}</td><td>{ex.get('gt_length','N/A')}</td></tr>\n"
            html += "</table>\n"
        else:
            html += f"<h3>{category.replace('_', ' ').title()}</h3>\n<p>No examples in this category.</p>\n"


    html += """
</body>
</html>
    """

    # Write report
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info("HTML report generated successfully.")
    except Exception as e:
        logger.error(f"Failed to write HTML report: {e}")