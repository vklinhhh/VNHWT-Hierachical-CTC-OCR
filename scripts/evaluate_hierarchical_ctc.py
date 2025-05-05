# scripts/evaluate_hierarchical_ctc.py
import os
import sys
import argparse
import torch
from datasets import load_dataset
import logging
import pandas as pd
from tqdm.auto import tqdm
import json
import itertools
import torch.nn as nn
# --- Imports from project ---
from model.hierarchical_ctc_model import HierarchicalCtcOcrModel # <<< USE HIERARCHICAL MODEL
from data.ctc_ocr_dataset import CtcOcrDataset # Standard CTC dataset
from data.ctc_collation import ctc_collate_fn # Standard CTC collate
from utils.ctc_utils import CTCDecoder, build_ctc_vocab # Standard CTC utils
from utils.evaluation_reporter import ( # Standard reporter
    calculate_corpus_metrics, analyze_errors, generate_visualizations, create_html_report
)
from torch.utils.data import DataLoader
from torch.nn import CTCLoss

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('evaluate_hierarchical_ctc.log')])
logger = logging.getLogger('EvaluateHierarchicalCtcScript')


def run_hierarchical_evaluation( # Renamed function
    model_path,
    dataset_name,
    output_dir,
    # Require COMBINED vocab path
    combined_char_vocab_path,
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
    try:
        logger.info(f"Loading combined char vocab from: {combined_char_vocab_path}")
        with open(combined_char_vocab_path, 'r', encoding='utf-8') as f: combined_char_vocab = json.load(f)
        if not combined_char_vocab: raise ValueError("Combined vocab empty.")
        combined_char_to_idx = {c: i for i, c in enumerate(combined_char_vocab)}
        combined_idx_to_char = {i: c for i, c in enumerate(combined_char_vocab)}
        # Determine blank index (should be 0)
        blank_idx = combined_char_to_idx.get('<blank>', 0)
        if blank_idx != 0: logger.warning("Blank token is not at index 0!")
        logger.info(f"Combined vocab loaded: {len(combined_char_vocab)} chars.")
    except Exception as e: logger.error(f"FATAL: Vocab load fail: {e}", exc_info=True); return


    # --- Load Model and Processor ---
    try:
        logger.info(f"Loading trained Hierarchical CTC model from: {model_path}")
        # Pass vocabs just in case config is missing, though combined is the crucial one
        # Base/Diac vocabs are defined in the model's config, not passed directly here usually
        model = HierarchicalCtcOcrModel.from_pretrained(
            model_path,
            combined_char_vocab=combined_char_vocab # Ensure config has this
            # Base/Diac vocabs should be loaded from config within the model if saved correctly
        )
        processor = model.processor
        model.to(device)
        model.eval()
        logger.info("Model and processor loaded.")
    except Exception as e: logger.error(f"FATAL: Model load fail: {e}", exc_info=True); return


    # --- Load Test Dataset ---
    # ... (same dataset loading logic as evaluate_ctc.py) ...
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
        # Check for required 'label' column for combined chars
        if 'label' not in test_hf_split.column_names and 'word' not in test_hf_split.column_names and 'text' not in test_hf_split.column_names:
             raise ValueError("Dataset needs 'label', 'word', or 'text' column for ground truth.")
    except Exception as dataset_load_e: logger.error(f"FATAL: Dataset load failed: {dataset_load_e}", exc_info=True); return


    # --- Create Dataset and DataLoader ---
    try:
        logger.info("Creating CTC test dataset wrapper (using combined vocab)...")
        # Use the standard CTC dataset with the COMBINED vocab map
        test_dataset = CtcOcrDataset(test_hf_split, processor, combined_char_to_idx, unk_token='[UNK]')
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=ctc_collate_fn, num_workers=num_workers,
            pin_memory=(device.type == 'cuda')
        )
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
    logger.info("Starting evaluation loop...")
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating Test Set")
        for batch in progress_bar:
            if batch is None: continue
            try:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'] # COMBINED char indices (CPU for loss)
                label_lengths = batch['label_lengths'] # CPU for loss
                texts_gt = batch['texts'] # Original text strings
                current_batch_size = pixel_values.size(0)

                # Forward pass -> uses model.forward
                outputs = model(pixel_values=pixel_values)
                # Use the FINAL logits for decoding and loss
                logits = outputs.get('logits') # Shape: [B, T, N_combined]

                if logits is None: continue

                # Calculate Loss (using final combined logits)
                log_probs = logits.log_softmax(2).permute(1, 0, 2) # T, B, C
                time_steps = log_probs.size(0)
                input_lengths = torch.full((current_batch_size,), time_steps, dtype=torch.long, device='cpu')
                input_lengths_clamped = torch.clamp(input_lengths, max=time_steps)
                label_lengths_clamped = torch.clamp(label_lengths, max=labels.size(1))
                try:
                    loss = ctc_loss_fn_eval(log_probs, labels, input_lengths_clamped, label_lengths_clamped)
                    total_loss_eval += loss.item()
                except Exception as loss_e: logger.warning(f"Loss calc error: {loss_e}")

                # Decode predictions using the FINAL combined logits
                decoded_preds = ctc_decoder(logits) # Decodes combined chars

                all_predictions.extend(decoded_preds)
                all_ground_truths.extend(texts_gt)

                batch_count_eval += 1

            except Exception as eval_batch_e:
                logger.error(f"Error during eval batch: {eval_batch_e}", exc_info=True)
                continue

    logger.info("Evaluation loop finished.")

    # --- Process Results ---
    if not all_ground_truths: logger.error("No samples processed."); return

    df_results = pd.DataFrame({'GroundTruth': all_ground_truths, 'Prediction': all_predictions})
    results_csv_path = os.path.join(output_dir, "evaluation_results_raw.csv")
    df_results.to_csv(results_csv_path, index=False)
    logger.info(f"Saved detailed results to: {results_csv_path}")

    # Calculate final metrics
    num_samples = len(df_results)
    avg_loss = total_loss_eval / num_samples if num_samples > 0 else 0.0
    cer, wer = calculate_corpus_metrics(df_results['Prediction'].tolist(), df_results['GroundTruth'].tolist())

    logger.info(f"Final Evaluation Metrics:")
    logger.info(f"  Average Loss: {avg_loss:.4f}")
    logger.info(f"  CER         : {cer:.4f}")
    logger.info(f"  WER         : {wer:.4f}")

    # --- Generate Report ---
    summary_stats = {
        'model_path': model_path, 'dataset_name': dataset_name, 'dataset_split': dataset_split,
        'total_samples': len(df_results), 'avg_loss': avg_loss,
        'cer': cer, 'wer': wer, # Report final CER/WER
    }
    # Use standard reporter, which now takes the prediction column (combined chars)
    error_analysis = analyze_errors(df_results) # Pass df with 'Prediction' column
    viz_dir = generate_visualizations(df_results, output_dir) # Pass df with 'Prediction'
    report_path = os.path.join(output_dir, "evaluation_report.html")
    create_html_report(report_path, summary_stats, error_analysis, viz_dir) # Pass adapted results

    logger.info(f"Evaluation complete. Report generated at {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Hierarchical CTC OCR model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained Hierarchical CTC model directory")
    parser.add_argument("--combined_char_vocab_path", type=str, required=True, help="Path to COMBINED char vocab JSON (e.g., combined_char_vocab.json)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Test dataset name/path.")
    parser.add_argument("--dataset_split", type=str, default="test", help="Test dataset split.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_hierarchical_ctc", help="Output directory.")
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