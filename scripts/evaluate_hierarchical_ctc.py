# scripts/evaluate_hierarchical_ctc.py
import os
import sys
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
import logging
import pandas as pd
from tqdm.auto import tqdm
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# --- Imports from project ---
# <<< CHANGE: Import the multi-scale hierarchical model >>>
from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
from data.ctc_ocr_dataset import CtcOcrDataset
from data.ctc_collation import ctc_collate_fn
from utils.ctc_utils import CTCDecoder
# Use the evaluation reporter functions
from utils.evaluation_reporter import (
    calculate_corpus_metrics, analyze_errors, generate_visualizations, create_html_report
)
from torch.utils.data import DataLoader
from torch.nn import CTCLoss

# --- Logging Setup ---
# ... (logging setup remains the same) ...
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('evaluate_hierarchical_ctc.log')])
logger = logging.getLogger('EvaluateHierarchicalCtcScript')


# --- New: Helper Functions for Diacritic Analysis ---
def extract_diacritics(text):
    """
    Extract diacritics from text for analysis.
    Returns a dictionary mapping character position to diacritic type.
    """
    diacritics_map = {}
    
    # Define dictionaries mapping characters to their base and diacritic components
    # This is a simplified approach - a complete solution would need a more comprehensive mapping
    diacritics = {
        # Acute accent (sắc)
        'á': ('a', 'acute'), 'é': ('e', 'acute'), 'í': ('i', 'acute'), 
        'ó': ('o', 'acute'), 'ú': ('u', 'acute'), 'ý': ('y', 'acute'),
        'Á': ('A', 'acute'), 'É': ('E', 'acute'), 'Í': ('I', 'acute'),
        'Ó': ('O', 'acute'), 'Ú': ('U', 'acute'), 'Ý': ('Y', 'acute'),
        
        # Grave accent (huyền)
        'à': ('a', 'grave'), 'è': ('e', 'grave'), 'ì': ('i', 'grave'),
        'ò': ('o', 'grave'), 'ù': ('u', 'grave'), 'ỳ': ('y', 'grave'),
        'À': ('A', 'grave'), 'È': ('E', 'grave'), 'Ì': ('I', 'grave'),
        'Ò': ('O', 'grave'), 'Ù': ('U', 'grave'), 'Ỳ': ('Y', 'grave'),
        
        # Hook accent (hỏi)
        'ả': ('a', 'hook'), 'ẻ': ('e', 'hook'), 'ỉ': ('i', 'hook'),
        'ỏ': ('o', 'hook'), 'ủ': ('u', 'hook'), 'ỷ': ('y', 'hook'),
        'Ả': ('A', 'hook'), 'Ẻ': ('E', 'hook'), 'Ỉ': ('I', 'hook'),
        'Ỏ': ('O', 'hook'), 'Ủ': ('U', 'hook'), 'Ỷ': ('Y', 'hook'),
        
        # Tilde (ngã)
        'ã': ('a', 'tilde'), 'ẽ': ('e', 'tilde'), 'ĩ': ('i', 'tilde'),
        'õ': ('o', 'tilde'), 'ũ': ('u', 'tilde'), 'ỹ': ('y', 'tilde'),
        'Ã': ('A', 'tilde'), 'Ẽ': ('E', 'tilde'), 'Ĩ': ('I', 'tilde'),
        'Õ': ('O', 'tilde'), 'Ũ': ('U', 'tilde'), 'Ỹ': ('Y', 'tilde'),
        
        # Dot below (nặng)
        'ạ': ('a', 'dot'), 'ẹ': ('e', 'dot'), 'ị': ('i', 'dot'),
        'ọ': ('o', 'dot'), 'ụ': ('u', 'dot'), 'ỵ': ('y', 'dot'),
        'Ạ': ('A', 'dot'), 'Ẹ': ('E', 'dot'), 'Ị': ('I', 'dot'),
        'Ọ': ('O', 'dot'), 'Ụ': ('U', 'dot'), 'Ỵ': ('Y', 'dot'),
        
        # Circumflex (mũ) + tone
        'ấ': ('a', 'circumflex_acute'), 'ầ': ('a', 'circumflex_grave'), 
        'ẩ': ('a', 'circumflex_hook'), 'ẫ': ('a', 'circumflex_tilde'), 
        'ậ': ('a', 'circumflex_dot'),
        'Ấ': ('A', 'circumflex_acute'), 'Ầ': ('A', 'circumflex_grave'), 
        'Ẩ': ('A', 'circumflex_hook'), 'Ẫ': ('A', 'circumflex_tilde'), 
        'Ậ': ('A', 'circumflex_dot'),
        
        'ế': ('e', 'circumflex_acute'), 'ề': ('e', 'circumflex_grave'), 
        'ể': ('e', 'circumflex_hook'), 'ễ': ('e', 'circumflex_tilde'), 
        'ệ': ('e', 'circumflex_dot'),
        'Ế': ('E', 'circumflex_acute'), 'Ề': ('E', 'circumflex_grave'), 
        'Ể': ('E', 'circumflex_hook'), 'Ễ': ('E', 'circumflex_tilde'), 
        'Ệ': ('E', 'circumflex_dot'),
        
        'ố': ('o', 'circumflex_acute'), 'ồ': ('o', 'circumflex_grave'), 
        'ổ': ('o', 'circumflex_hook'), 'ỗ': ('o', 'circumflex_tilde'), 
        'ộ': ('o', 'circumflex_dot'),
        'Ố': ('O', 'circumflex_acute'), 'Ồ': ('O', 'circumflex_grave'), 
        'Ổ': ('O', 'circumflex_hook'), 'Ỗ': ('O', 'circumflex_tilde'), 
        'Ộ': ('O', 'circumflex_dot'),
        
        # Breve (trăng) + tone
        'ắ': ('a', 'breve_acute'), 'ằ': ('a', 'breve_grave'), 
        'ẳ': ('a', 'breve_hook'), 'ẵ': ('a', 'breve_tilde'), 
        'ặ': ('a', 'breve_dot'),
        'Ắ': ('A', 'breve_acute'), 'Ằ': ('A', 'breve_grave'), 
        'Ẳ': ('A', 'breve_hook'), 'Ẵ': ('A', 'breve_tilde'), 
        'Ặ': ('A', 'breve_dot'),
        
        # Horn (móc) + tone
        'ớ': ('o', 'horn_acute'), 'ờ': ('o', 'horn_grave'), 
        'ở': ('o', 'horn_hook'), 'ỡ': ('o', 'horn_tilde'), 
        'ợ': ('o', 'horn_dot'),
        'Ớ': ('O', 'horn_acute'), 'Ờ': ('O', 'horn_grave'), 
        'Ở': ('O', 'horn_hook'), 'Ỡ': ('O', 'horn_tilde'), 
        'Ợ': ('O', 'horn_dot'),
        
        'ứ': ('u', 'horn_acute'), 'ừ': ('u', 'horn_grave'), 
        'ử': ('u', 'horn_hook'), 'ữ': ('u', 'horn_tilde'), 
        'ự': ('u', 'horn_dot'),
        'Ứ': ('U', 'horn_acute'), 'Ừ': ('U', 'horn_grave'), 
        'Ử': ('U', 'horn_hook'), 'Ữ': ('U', 'horn_tilde'), 
        'Ự': ('U', 'horn_dot'),
        
        # Circumflex/breve/horn without tone (just the modifier)
        'â': ('a', 'circumflex'), 'ê': ('e', 'circumflex'), 'ô': ('o', 'circumflex'),
        'Â': ('A', 'circumflex'), 'Ê': ('E', 'circumflex'), 'Ô': ('O', 'circumflex'),
        
        'ă': ('a', 'breve'), 'Ă': ('A', 'breve'),
        
        'ơ': ('o', 'horn'), 'ư': ('u', 'horn'),
        'Ơ': ('O', 'horn'), 'Ư': ('U', 'horn'),
        
        # Stroke (đ)
        'đ': ('d', 'stroke'), 'Đ': ('D', 'stroke'),
    }
    
    # Process each character
    for i, char in enumerate(text):
        if char in diacritics:
            base_char, diacritic = diacritics[char]
            diacritics_map[i] = {'char': char, 'base': base_char, 'diacritic': diacritic}
    
    return diacritics_map


def analyze_diacritic_accuracy(ground_truths, predictions):
    """
    Analyze diacritic prediction accuracy in detail.
    """
    diacritic_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'examples': []})
    
    # Process each pair of ground truth and prediction
    for gt, pred in zip(ground_truths, predictions):
        # Skip if lengths don't match (likely a major recognition error)
        if len(gt) != len(pred):
            continue
        
        # Extract diacritics from ground truth and prediction
        gt_diacritics = extract_diacritics(gt)
        pred_diacritics = extract_diacritics(pred)
        
        # Analyze diacritic prediction accuracy
        for i in gt_diacritics:
            if i < len(pred):  # Ensure position exists in prediction
                gt_diac = gt_diacritics[i]['diacritic']
                diacritic_stats[gt_diac]['total'] += 1
                
                # Check if prediction has same position and diacritic
                if i in pred_diacritics and pred_diacritics[i]['diacritic'] == gt_diac:
                    diacritic_stats[gt_diac]['correct'] += 1
                else:
                    # Record example of error
                    error_example = {
                        'gt_char': gt_diacritics[i]['char'],
                        'pred_char': pred[i] if i < len(pred) else '',
                        'gt_context': gt[max(0, i-5):min(len(gt), i+6)],
                        'pred_context': pred[max(0, i-5):min(len(pred), i+6)],
                    }
                    if len(diacritic_stats[gt_diac]['examples']) < 10:  # Limit to 10 examples
                        diacritic_stats[gt_diac]['examples'].append(error_example)
    
    # Calculate accuracy for each diacritic type
    results = {}
    for diac, stats in diacritic_stats.items():
        if stats['total'] > 0:
            results[diac] = {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total'],
                'examples': stats['examples']
            }
    
    return results


def visualize_diacritic_accuracy(diacritic_results, output_dir):
    """
    Create visualizations for diacritic prediction accuracy.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    diacritics = []
    accuracies = []
    counts = []
    
    for diac, stats in diacritic_results.items():
        diacritics.append(diac)
        accuracies.append(stats['accuracy'] * 100)  # Convert to percentage
        counts.append(stats['total'])
    
    # Sort by frequency
    sorted_indices = np.argsort(counts)[::-1]
    diacritics = [diacritics[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    # Create accuracy bar chart
    plt.figure(figsize=(14, 8))
    bars = plt.bar(diacritics, accuracies, color='skyblue')
    
    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 str(count), ha='center', va='bottom', rotation=0, 
                 fontsize=9, color='black')
    
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='--', 
                label=f'Average: {np.mean(accuracies):.1f}%')
    
    plt.xlabel('Diacritic Type')
    plt.ylabel('Accuracy (%)')
    plt.title('Diacritic Recognition Accuracy by Type')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)  # Leave room for count labels
    plt.tight_layout()
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'diacritic_accuracy.png'))
    plt.close()
    
    # Create a heatmap of confusion between commonly confused diacritics
    # This would require more detailed analysis of error cases
    
    return os.path.join(output_dir, 'diacritic_accuracy.png')


def run_hierarchical_evaluation( # Renamed function
    model_path,
    dataset_name,
    output_dir,
    combined_char_vocab_path, # Requires combined vocab
    dataset_split='test',
    batch_size=16,
    num_workers=4,
    device=None
    ):
    """Runs evaluation on the test dataset for the Hierarchical model."""

    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Combined Vocabulary FIRST ---
    # ... (vocab loading remains the same) ...
    try:
        logger.info(f"Loading combined char vocab from: {combined_char_vocab_path}")
        with open(combined_char_vocab_path, 'r', encoding='utf-8') as f: combined_char_vocab = json.load(f)
        if not combined_char_vocab: raise ValueError("Combined vocab empty.")
        combined_char_to_idx = {c: i for i, c in enumerate(combined_char_vocab)}
        combined_idx_to_char = {i: c for i, c in enumerate(combined_char_vocab)}
        blank_idx = combined_char_to_idx.get('<blank>', 0)
        logger.info(f"Combined vocab loaded: {len(combined_char_vocab)} chars.")
    except Exception as e: logger.error(f"FATAL: Vocab load fail: {e}", exc_info=True); return


    # --- Load Model and Processor ---
    try:
        logger.info(f"Loading trained Hierarchical CTC model from: {model_path}")
        # <<< CHANGE: Use the correct model class name >>>
        model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(
            model_path,
            combined_char_vocab=combined_char_vocab # Pass vocab in case config is missing/minimal
            # Other vocabs (base/diac) should be in config if saved correctly
        )
        processor = model.processor
        model.to(device)
        model.eval()
        logger.info("Model and processor loaded.")
        
        # Log which diacritic enhancement modules are enabled
        if hasattr(model.config, 'use_visual_diacritic_attention') and model.config.use_visual_diacritic_attention:
            logger.info("Model uses Visual Diacritic Attention")
        if hasattr(model.config, 'use_character_diacritic_compatibility') and model.config.use_character_diacritic_compatibility:
            logger.info("Model uses Character-Diacritic Compatibility Matrix")
        if hasattr(model.config, 'use_few_shot_diacritic_adapter') and model.config.use_few_shot_diacritic_adapter:
            logger.info(f"Model uses Few-Shot Diacritic Adapter with {model.config.num_few_shot_prototypes} prototypes")
        
    except Exception as e:
        logger.error(f"FATAL: Failed to load model: {e}", exc_info=True); return

    # --- Load Test Dataset ---
    # ... (dataset loading logic remains the same) ...
    try:
        logger.info(f"Loading test dataset: {dataset_name}, split: {dataset_split}")
        hf_dataset = load_dataset(dataset_name)
        if dataset_split not in hf_dataset:
             if 'test' in hf_dataset: dataset_split = 'test'
             elif 'validation' in hf_dataset: dataset_split = 'validation'
             else: raise ValueError(f"Split '{dataset_split}' not found.")
             logger.warning(f"Using dataset split: '{dataset_split}'")
        test_hf_split = hf_dataset[dataset_split]
        logger.info(f"Test set size: {len(test_hf_split)}")
        if not any(col in test_hf_split.column_names for col in ['label', 'word', 'text']):
             raise ValueError("Dataset needs 'label', 'word', or 'text' column.")
    except Exception as dataset_load_e: logger.error(f"FATAL: Dataset load failed: {dataset_load_e}", exc_info=True); return


    # --- Create Dataset and DataLoader ---
    # ... (dataset creation remains the same, uses combined map) ...
    try:
        logger.info("Creating CTC test dataset wrapper (using combined vocab)...")
        test_dataset = CtcOcrDataset(test_hf_split, processor, combined_char_to_idx, unk_token='[UNK]')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=ctc_collate_fn, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
        logger.info(f"Test DataLoader created.")
    except Exception as dataset_wrap_e: logger.error(f"FATAL: Dataset/loader failed: {dataset_wrap_e}", exc_info=True); return


    # --- Initialize Decoder and Storage ---
    # Use decoder based on the COMBINED vocabulary
    ctc_decoder = CTCDecoder(idx_to_char_map=combined_idx_to_char, blank_idx=blank_idx)
    all_predictions = []
    all_ground_truths = []
    total_loss_eval = 0.0
    batch_count_eval = 0
    ctc_loss_fn_eval = nn.CTCLoss(blank=blank_idx, reduction='sum', zero_infinity=True)

    # --- Run Inference Loop ---
    # ... (Inference loop remains the same - uses model.forward(), gets final 'logits') ...
    logger.info("Starting evaluation loop...")
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating Test Set")
        for batch in progress_bar:
            if batch is None: continue
            try:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'] # Keep CPU for loss
                label_lengths = batch['label_lengths'] # Keep CPU for loss
                texts_gt = batch['texts'] # Original text strings
                current_batch_size = pixel_values.size(0)

                outputs = model(pixel_values=pixel_values)
                logits = outputs.get('logits') # Get final combined logits

                if logits is None: continue

                # Calculate Loss
                log_probs = logits.log_softmax(2).permute(1, 0, 2)
                time_steps = log_probs.size(0)
                input_lengths = torch.full((current_batch_size,), time_steps, dtype=torch.long, device='cpu')
                input_lengths_clamped = torch.clamp(input_lengths, max=time_steps)
                label_lengths_clamped = torch.clamp(label_lengths, max=labels.size(1))
                try: loss = ctc_loss_fn_eval(log_probs, labels, input_lengths_clamped, label_lengths_clamped); total_loss_eval += loss.item()
                except Exception as loss_e: logger.warning(f"Loss calc error: {loss_e}")

                # Decode predictions
                decoded_preds = ctc_decoder(logits) # Decode combined chars

                all_predictions.extend(decoded_preds)
                all_ground_truths.extend(texts_gt)

                batch_count_eval += 1

            except Exception as eval_batch_e: logger.error(f"Error eval batch: {eval_batch_e}", exc_info=True); continue

    logger.info("Evaluation loop finished.")

    # --- Process Results (Uses standard CTC metrics) ---
    # ... (Result processing and reporting remain the same) ...
    if not all_ground_truths: logger.error("No samples processed."); return
    df_results = pd.DataFrame({'GroundTruth': all_ground_truths, 'Prediction': all_predictions})
    results_csv_path = os.path.join(output_dir, "evaluation_results_raw.csv")
    df_results.to_csv(results_csv_path, index=False)
    logger.info(f"Saved detailed results to: {results_csv_path}")
    num_samples = len(df_results)
    avg_loss = total_loss_eval / num_samples if num_samples > 0 else 0.0
    cer, wer = calculate_corpus_metrics(df_results['Prediction'].tolist(), df_results['GroundTruth'].tolist())
    logger.info(f"Final Evaluation Metrics:"); logger.info(f"  Average Loss: {avg_loss:.4f}"); logger.info(f"  CER         : {cer:.4f}"); logger.info(f"  WER         : {wer:.4f}")
    
    # --- NEW: Analyze Diacritic Accuracy ---
    logger.info("Analyzing diacritic recognition accuracy...")
    diacritic_results = analyze_diacritic_accuracy(df_results['GroundTruth'].tolist(), df_results['Prediction'].tolist())
    
    # Generate diacritic visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    diacritic_viz_path = visualize_diacritic_accuracy(diacritic_results, viz_dir)
    
    # Print diacritic accuracy summary
    logger.info("Diacritic Recognition Accuracy:")
    for diac, stats in sorted(diacritic_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        logger.info(f"  {diac}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
    
    # Save diacritic analysis to JSON
    diacritic_json_path = os.path.join(output_dir, "diacritic_analysis.json")
    with open(diacritic_json_path, 'w', encoding='utf-8') as f:
        json.dump(diacritic_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved diacritic analysis to: {diacritic_json_path}")
    
    # --- Continue with standard evaluation reporting ---
    summary_stats = {'model_path': model_path, 'dataset_name': dataset_name, 'dataset_split': dataset_split, 'total_samples': len(df_results), 'avg_loss': avg_loss, 'cer': cer, 'wer': wer}
    error_analysis = analyze_errors(df_results)
    
    # Add diacritic info to summary stats
    diacritic_avg_accuracy = np.mean([stats['accuracy'] for stats in diacritic_results.values()])
    summary_stats['diacritic_avg_accuracy'] = diacritic_avg_accuracy
    
    # Standard visualization generation
    standard_viz_dir = generate_visualizations(df_results, output_dir)
    
    # Create HTML report
    report_path = os.path.join(output_dir, "evaluation_report.html")
    create_html_report(report_path, summary_stats, error_analysis, viz_dir)
    
    # Create additional HTML report specifically for diacritics
    diacritic_report_path = os.path.join(output_dir, "diacritic_evaluation_report.html")
    create_diacritic_html_report(diacritic_report_path, diacritic_results, model_path, diacritic_viz_path)
    
    logger.info(f"Evaluation complete. Reports generated at {report_path} and {diacritic_report_path}")


def create_diacritic_html_report(report_path, diacritic_results, model_path, viz_path):
    """Create an HTML report specifically for diacritic analysis."""
    logger.info(f"Generating diacritic HTML report at: {report_path}")
    
    # Calculate overall diacritic accuracy
    total_correct = sum(stats['correct'] for stats in diacritic_results.values())
    total_diacritics = sum(stats['total'] for stats in diacritic_results.values())
    overall_accuracy = total_correct / total_diacritics if total_diacritics > 0 else 0
    
    # Sort diacritics by accuracy (descending)
    sorted_diacritics = sorted(
        diacritic_results.items(), 
        key=lambda x: (x[1]['accuracy'], -x[1]['total']), 
        reverse=True
    )
    
    # Basic HTML structure
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Hierarchical CTC OCR: Diacritic Evaluation Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary-box {{ border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; background-color: #f9f9f9; }}
        .metrics td {{ font-weight: bold; color: #0056b3; }}
        .high {{ color: green; }}
        .medium {{ color: #cc7000; }}
        .low {{ color: red; }}
        img {{ max-width: 800px; height: auto; display: block; margin: 20px auto; border: 1px solid #ccc; }}
        .examples {{ font-family: monospace; background-color: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Hierarchical CTC OCR: Diacritic Evaluation Report</h1>
    <p>Model Path: {model_path}</p>

    <h2>Summary</h2>
    <div class="summary-box metrics">
        <table>
            <tr><td>Total Diacritics Evaluated</td><td>{total_diacritics}</td></tr>
            <tr><td>Overall Diacritic Accuracy</td><td>{overall_accuracy:.4f} ({total_correct}/{total_diacritics})</td></tr>
            <tr><td>Number of Diacritic Types</td><td>{len(diacritic_results)}</td></tr>
        </table>
    </div>

    <h2>Diacritic Accuracy Visualization</h2>
    <img src="{os.path.basename(viz_path)}" alt="Diacritic Accuracy Chart">

    <h2>Diacritic Type Breakdown</h2>
    <table>
        <tr>
            <th>Diacritic</th>
            <th>Accuracy</th>
            <th>Correct</th>
            <th>Total</th>
            <th>Example Errors</th>
        </tr>
    """
    
    # Add rows for each diacritic type
    for diac, stats in sorted_diacritics:
        accuracy = stats['accuracy']
        accuracy_class = 'high' if accuracy >= 0.9 else ('medium' if accuracy >= 0.7 else 'low')
        
        # Format example errors
        example_html = ""
        if stats['examples']:
            for i, example in enumerate(stats['examples'][:3]):  # Show up to 3 examples
                example_html += f"""
                <div class="examples">
                    <strong>Error {i+1}:</strong><br>
                    GT: {example['gt_char']} (in "{example['gt_context']}")<br>
                    Pred: {example['pred_char']} (in "{example['pred_context']}")
                </div>
                """
        else:
            example_html = "<em>No errors found</em>"
        
        html += f"""
        <tr>
            <td>{diac}</td>
            <td class="{accuracy_class}">{accuracy:.4f} ({accuracy*100:.1f}%)</td>
            <td>{stats['correct']}</td>
            <td>{stats['total']}</td>
            <td>{example_html}</td>
        </tr>
        """
    
    html += """
    </table>

    <h2>Analysis and Recommendations</h2>
    <div class="summary-box">
        <h3>Diacritic Categories Performance</h3>
    """
    
    # Group diacritics by category and calculate performance
    categories = {
        'tone_marks': ['acute', 'grave', 'hook', 'tilde', 'dot'],
        'modifiers': ['circumflex', 'breve', 'horn', 'stroke'],
        'combined': [d for d in diacritic_results.keys() if '_' in d]
    }
    
    category_stats = {}
    for category, diacritics in categories.items():
        category_diacritics = [d for d in diacritics if d in diacritic_results]
        if category_diacritics:
            correct = sum(diacritic_results[d]['correct'] for d in category_diacritics)
            total = sum(diacritic_results[d]['total'] for d in category_diacritics)
            accuracy = correct / total if total > 0 else 0
            category_stats[category] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
    
    # Add category performance to report
    for category, stats in category_stats.items():
        category_display = category.replace('_', ' ').title()
        accuracy_class = 'high' if stats['accuracy'] >= 0.9 else ('medium' if stats['accuracy'] >= 0.7 else 'low')
        
        html += f"""
        <p><strong>{category_display}:</strong> <span class="{accuracy_class}">{stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})</span></p>
        """
    
    # Add recommendations based on results
    html += """
        <h3>Key Findings and Recommendations</h3>
        <ul>
    """
    
    # Find problematic diacritics (accuracy < 0.8 and at least 10 samples)
    problem_diacritics = [(d, stats) for d, stats in diacritic_results.items() 
                          if stats['accuracy'] < 0.8 and stats['total'] >= 10]
    
    if problem_diacritics:
        html += "<li><strong>Problematic Diacritics:</strong><ul>"
        for diac, stats in sorted(problem_diacritics, key=lambda x: x[1]['accuracy']):
            html += f"""
            <li><strong>{diac}</strong>: Accuracy {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})
            <br>Recommendation: Consider enhancing the model's ability to recognize this diacritic.</li>
            """
        html += "</ul></li>"
    
    # Check if combined diacritics perform worse than simple ones
    simple_diacritics = [d for d in diacritic_results.keys() if '_' not in d and d != 'no_diacritic']
    combined_diacritics = [d for d in diacritic_results.keys() if '_' in d]
    
    if simple_diacritics and combined_diacritics:
        simple_acc = sum(diacritic_results[d]['correct'] for d in simple_diacritics) / \
                     sum(diacritic_results[d]['total'] for d in simple_diacritics)
        combined_acc = sum(diacritic_results[d]['correct'] for d in combined_diacritics) / \
                       sum(diacritic_results[d]['total'] for d in combined_diacritics)
        
        if combined_acc < simple_acc - 0.1:  # If combined is notably worse
            html += f"""
            <li><strong>Combined vs. Simple Diacritics:</strong> Combined diacritics have significantly lower accuracy 
            ({combined_acc:.4f}) than simple diacritics ({simple_acc:.4f}).
            <br>Recommendation: Consider enhancing the model's capability with combined diacritics, potentially through:
            <ul>
                <li>Adding more training samples with combined diacritics</li>
                <li>Using the Few-Shot Diacritic Adapter for rare combinations</li>
                <li>Fine-tuning the Visual Diacritic Attention mechanism</li>
            </ul>
            </li>
            """
    
    # Add general recommendations based on performance
    overall_recommendation = ""
    if overall_accuracy < 0.7:
        overall_recommendation = """
        <li><strong>General Performance:</strong> Overall diacritic accuracy is low.
        <br>Recommendation: Consider fundamental changes to the diacritic classifier, such as:
        <ul>
            <li>Enable all three diacritic enhancement modules</li>
            <li>Increase training data with diverse diacritical marks</li>
            <li>Consider a specialized training phase focusing only on diacritics</li>
        </ul>
        </li>
        """
    elif overall_accuracy < 0.9:
        overall_recommendation = """
        <li><strong>General Performance:</strong> Overall diacritic accuracy is moderate.
        <br>Recommendation: Some targeted improvements could help:
        <ul>
            <li>Enable Visual Diacritic Attention to better focus on diacritic regions</li>
            <li>Add more examples of the lower-performing diacritics to the training set</li>
            <li>Use Character-Diacritic Compatibility to improve linguistic consistency</li>
        </ul>
        </li>
        """
    else:
        overall_recommendation = """
        <li><strong>General Performance:</strong> Overall diacritic accuracy is excellent.
        <br>Recommendation: Focus on the few problematic diacritics if any.</li>
        """
    
    html += overall_recommendation
    
    html += """
        </ul>
    </div>
</body>
</html>
    """
    
    # Write report
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info("Diacritic HTML report generated successfully.")
    except Exception as e:
        logger.error(f"Failed to write diacritic HTML report: {e}")


if __name__ == "__main__":
    # ... (Argument parsing remains the same as evaluate_ctc.py) ...
    parser = argparse.ArgumentParser(description="Evaluate a trained Hierarchical CTC OCR model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained Hierarchical CTC model directory")
    parser.add_argument("--combined_char_vocab_path", type=str, required=True, help="Path to COMBINED char vocab JSON")
    parser.add_argument("--dataset_name", type=str, required=True, help="Test dataset name/path.")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_hierarchical_ctc")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    run_hierarchical_evaluation( # Call the evaluation function
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        combined_char_vocab_path=args.combined_char_vocab_path, # Pass combined vocab path
        dataset_split=args.dataset_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )