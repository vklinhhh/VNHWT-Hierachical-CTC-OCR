# training/ctc_validation.py
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import logging
# Optional: pip install torchmetrics editdistance
# from torchmetrics.text import CharErrorRate, WordErrorRate
import editdistance # For calculating CER/WER manually if not using torchmetrics

logger = logging.getLogger(__name__)

def compute_ctc_validation_metrics(model, val_loader, device, ctc_decoder):
    """
    Compute validation metrics for the CTC OCR model.
    Includes loss, Character Error Rate (CER), Word Error Rate (WER),
    and logs sample predictions.
    """
    model.eval()
    total_val_loss = 0.0
    batch_count = 0

    all_preds_strings = [] # Store decoded strings
    all_gts_strings = []   # Store ground truth strings

    # Flag to control sample logging (only first batch)
    logged_samples_count = 0
    max_samples_to_log = 5 # How many samples to log

    # CTC Loss (re-instantiated for validation)
    # Use reduction='sum' to manually average later by number of samples or characters
    ctc_loss_fn = nn.CTCLoss(blank=ctc_decoder.blank_idx, reduction='sum', zero_infinity=True)

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None: continue

            try:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device) # Padded char indices [B, MaxLabelLen]
                label_lengths = batch['label_lengths'].to(device) # Actual lengths [B]
                texts_gt = batch['texts'] # Original text strings list[str]
                current_batch_size = pixel_values.size(0)
            except Exception as move_e:
                logger.error(f"Error moving validation batch {batch_idx} to device: {move_e}", exc_info=True)
                continue

            try:
                # Forward pass
                outputs = model(pixel_values=pixel_values) # Get only logits
                logits = outputs.get('logits') # Shape: [B, T, C]

                if logits is None:
                     logger.warning(f"Logits missing for batch {batch_idx}. Skipping.")
                     continue

                # --- Calculate Loss ---
                log_probs = logits.log_softmax(2).permute(1, 0, 2) # T, B, C
                time_steps = log_probs.size(0)
                input_lengths = torch.full((current_batch_size,), time_steps, dtype=torch.long, device='cpu') # CPU for loss fn
                labels_cpu = labels.cpu()
                label_lengths_cpu = label_lengths.cpu()

                # Clamp lengths
                input_lengths_clamped = torch.clamp(input_lengths, max=time_steps)
                label_lengths_clamped = torch.clamp(label_lengths_cpu, max=labels_cpu.size(1))

                loss = ctc_loss_fn(log_probs, labels_cpu, input_lengths_clamped, label_lengths_clamped)
                total_val_loss += loss.item() # Accumulate sum loss

                # --- Decode Predictions ---
                # Use the provided ctc_decoder (e.g., greedy)
                decoded_preds = ctc_decoder(logits) # Expects [B, T, C], returns list[str]

                # Store predictions and ground truths for corpus-level metrics later
                all_preds_strings.extend(decoded_preds)
                all_gts_strings.extend(texts_gt)

                # --- Log Sample Predictions (First Batch Only) ---
                if batch_idx == 0:
                    num_to_log = min(max_samples_to_log - logged_samples_count, current_batch_size)
                    for i in range(num_to_log):
                        logger.info(f"--- Validation Sample {logged_samples_count + i} ---")
                        logger.info(f"  Ground Truth: '{texts_gt[i]}'")
                        logger.info(f"  Prediction  : '{decoded_preds[i]}'")
                        logger.info(f"---------------------------------")
                    logged_samples_count += num_to_log


                batch_count += 1

            except Exception as batch_e:
                logger.error(f"Error processing validation batch {batch_idx}: {batch_e}", exc_info=True)
                # Log shapes on error for debugging
                logger.error(f"  Logits shape: {logits.shape if 'logits' in locals() else 'N/A'}")
                logger.error(f"  Labels shape: {labels.shape}")
                logger.error(f"  Label lengths: {label_lengths.tolist()}")
                continue # Skip batch on error

    # --- End of Validation Loop ---
    model.train() # Set back to training mode

    # --- Calculate Final Metrics ---
    # Average Loss: Divide total sum loss by total number of samples
    avg_val_loss = total_val_loss / len(all_gts_strings) if all_gts_strings else 0.0

    # Calculate Corpus CER and WER using editdistance
    total_edit_distance_char = 0
    total_chars = 0
    total_edit_distance_word = 0
    total_words = 0

    if all_gts_strings: # Ensure we have samples
         for pred_str, gt_str in zip(all_preds_strings, all_gts_strings):
              # Guard against errors in editdistance if strings are weird
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


    val_cer = total_edit_distance_char / total_chars if total_chars > 0 else 1.0 # Default 1 if no chars
    val_wer = total_edit_distance_word / total_words if total_words > 0 else 1.0 # Default 1 if no words

    # Log final results
    logger.info(f"Validation Results: Loss={avg_val_loss:.4f}, CER={val_cer:.4f}, WER={val_wer:.4f}")
    logger.info(f"(Based on {len(all_gts_strings)} samples)")

    # Return metrics dictionary
    results = {
        'val_loss': avg_val_loss,
        'val_cer': val_cer,
        'val_wer': val_wer
    }
    return results