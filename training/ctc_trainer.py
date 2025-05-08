# training/ctc_trainer.py
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm.auto import tqdm
import logging
import gc
from torch.cuda.amp import GradScaler, autocast

# --- Update imports ---
# Note: We KEEP using compute_ctc_validation_metrics as the hierarchical model
# still outputs a single sequence of combined char logits for the final CTC loss/eval.
from .ctc_validation import compute_ctc_validation_metrics
# Use the standard CTC collate function as the dataset provides combined char labels
from data.ctc_collation import ctc_collate_fn
from utils.ctc_utils import CTCDecoder # For potential decoding during logging/eval
from utils.schedulers import CosineWarmupWithPlateauScheduler  # Import the combined scheduler

# Set up logging
logger = logging.getLogger('CtcTrainer') # Keep original logger name or change if preferred
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
    # Optional: Add file handler
    # fh = logging.FileHandler('ctc_training.log')
    # fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # logger.addHandler(fh)


# --- save_checkpoint function remains the same ---
def save_checkpoint(state, filepath, is_best):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    base_path = os.path.splitext(filepath)[0]
    latest_path = f"{base_path}_latest.pt"
    try:
        logger.info(f"Saving latest checkpoint to {latest_path}...")
        torch.save(state, latest_path)
        logger.info(f"Latest checkpoint saved successfully.")
        if is_best:
            best_path = f"{base_path}_best.pt"
            logger.info(f"Saving best checkpoint to {best_path}...")
            torch.save(state, best_path)
            logger.info(f"Best checkpoint saved successfully.")
    except Exception as e:
        logger.error(f"Error saving checkpoint to {filepath}: {e}", exc_info=True)

# --- Training function (can keep the same name, logic handles different models) ---
def train_ctc_model(
    model,                      # Can be CtcOcrModel or HierarchicalCtcOcrModel instance
    optimizer,
    lr_scheduler,
    train_dataset,              # CtcOcrDataset instance
    val_dataset,                # CtcOcrDataset instance
    # --- CTC Specific ---
    ctc_loss_weight=1.0,        # Weight (relevant if model returns multiple losses, not used here)
    # --- Core Training Parameters ---
    epochs=10,
    batch_size=8,
    device=None,
    output_dir="ctc_ocr_model", # Default, overridden by script
    # --- Resuming ---
    start_epoch=0,
    resumed_optimizer_steps=0,
    resumed_best_val_metric=float('inf'),
    best_metric_name='val_loss',
    # --- Logging & Saving ---
    project_name="ctc-viet-ocr", # Default, overridden by script
    run_name=None,
    log_interval=10,
    save_checkpoint_prefix="checkpoint",
    # --- Optimization ---
    use_amp=False,
    scaler_state_to_load=None,
    grad_accumulation_steps=1,
    # --- Dataloader ---
    num_workers=4,
    # --- Validation & Early Stopping ---
    eval_steps=None,
    early_stopping_patience=5,
    early_stopping_metric='val_loss',
):
    """
    Training function for CTC-based OCR models (single or hierarchical head).
    Uses standard CTC loss on the model's final 'logits' output.
    *** Updated to support ReduceLROnPlateau scheduler. ***
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Trainer using device: {device}")

    # --- Determine if higher value is better for metric ---
    higher_is_better = best_metric_name not in ['val_loss', 'val_cer', 'val_wer']

    # --- Identify which scheduler type we're using ---
    is_plateau_scheduler = isinstance(lr_scheduler, CosineWarmupWithPlateauScheduler)
    if is_plateau_scheduler:
        logger.info("Using CosineWarmupWithPlateauScheduler - will update based on validation metrics")
    
    # --- Setup Output Dirs & WandB ---
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    wandb_run = None
    if project_name:
        try:
            # Config includes args passed from the calling script
            wandb_config = {
                "total_epochs": epochs, "start_epoch": start_epoch, "batch_size": batch_size,
                "learning_rate": optimizer.param_groups[0]['lr'], "grad_accumulation_steps": grad_accumulation_steps,
                "use_amp": use_amp, "device": str(device),
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_metric": early_stopping_metric,
                "best_metric_name": best_metric_name,
                "num_workers": num_workers, "eval_steps": eval_steps, "log_interval": log_interval,
                "weight_decay": optimizer.defaults.get('weight_decay', None),
                "model_type": model.config.model_type, # Get from model's config
                "vocab_size": getattr(model.config, 'combined_char_vocab_size', getattr(model.config, 'vocab_size', None)),
            }
            wandb_run = wandb.init(project=project_name, name=run_name, resume="allow", config=wandb_config)
            logger.info(f"Initialized WandB run: {wandb_run.id if wandb_run else 'Failed'}")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            wandb_run = None

    # --- Setup DataLoaders ---
    pin_memory_flag = (device.type == 'cuda')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=ctc_collate_fn, # Use standard CTC collate function
        num_workers=num_workers, pin_memory=pin_memory_flag, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=ctc_collate_fn, # Use standard CTC collate function
        num_workers=num_workers, pin_memory=pin_memory_flag, persistent_workers=(num_workers > 0)
    )
    logger.info(f"Train DataLoader: {len(train_loader)} batches, Val DataLoader: {len(val_loader)} batches")

    # --- Initialize AMP Scaler ---
    scaler = GradScaler(enabled=use_amp)
    # *** Load state if provided ***
    if scaler_state_to_load and use_amp:
         try:
             scaler.load_state_dict(scaler_state_to_load)
             logger.info("Trainer successfully loaded previous GradScaler state.")
         except Exception as scaler_load_e:
              logger.warning(f"Trainer could not load GradScaler state: {scaler_load_e}. Starting with default scale.")
    elif use_amp:
         logger.info("Trainer starting with default GradScaler state.")
    # *** End Load State ***

    # --- Initialize Training State ---
    best_metric_value = resumed_best_val_metric if not higher_is_better else -resumed_best_val_metric
    no_improvement_count = 0
    optimizer_steps = resumed_optimizer_steps
    training_exception = None
    best_model_state_dict = None

    # --- Load Checkpoint State (includes extracting scaler state) ---
    # This block comes from the calling script's logic now, setting start_epoch etc.
    # We just need to handle the scaler_state_to_load extraction if use_amp=True
    # Assume start_epoch, resumed_optimizer_steps, resumed_best_val_metric are passed correctly
    # We need to retrieve scaler_state_to_load if resuming
    checkpoint_to_load = None # This would typically be determined in the main script
    # Simulate finding scaler state if resuming (in real script, this check happens there)
    if start_epoch > 0 and use_amp: # Indicates a resume with AMP possibly enabled before
        # Check if the checkpoint exists and load it temporarily to get scaler state
        potential_checkpoint_path = os.path.join(output_dir, "checkpoints", f"{save_checkpoint_prefix}_latest.pt")
        if os.path.exists(potential_checkpoint_path):
             try:
                 logger.info(f"Checking checkpoint {potential_checkpoint_path} for scaler state...")
                 checkpoint = torch.load(potential_checkpoint_path, map_location='cpu') # Load to CPU temporarily
                 if 'scaler_state_dict' in checkpoint:
                     scaler_state_to_load = checkpoint['scaler_state_dict']
                     logger.info("Found scaler state in checkpoint.")
                 else:
                     logger.warning("Resuming training, but scaler state not found in checkpoint.")
                 del checkpoint # Free memory
             except Exception as e:
                 logger.warning(f"Could not read checkpoint {potential_checkpoint_path} to check for scaler state: {e}")


    # --- Initialize CTC Decoder for Validation Logging ---
    validation_decoder = None
    try:
        if hasattr(model.config, 'combined_char_vocab') and model.config.combined_char_vocab:
            vocab_list = model.config.combined_char_vocab
            logger.info("Using combined_char_vocab for validation decoder.")
        elif hasattr(model.config, 'char_vocab') and model.config.char_vocab:
             vocab_list = model.config.char_vocab
             logger.info("Using char_vocab for validation decoder.")
        else: raise ValueError("Suitable character vocabulary not found in model config.")

        idx_to_char = {i: ch for i, ch in enumerate(vocab_list)}
        blank_idx = 0
        for i, char in idx_to_char.items():
             if char == '<blank>': blank_idx = i; break
        validation_decoder = CTCDecoder(idx_to_char_map=idx_to_char, blank_idx=blank_idx)
        logger.info(f"Validation CTCDecoder initialized (Blank Index: {blank_idx}).")
    except Exception as e:
        logger.error(f"Failed to initialize validation CTCDecoder: {e}. Validation sample logging disabled.")


    # ==========================================================
    # --- Main Training Loop ---
    # ==========================================================
    logger.info(f"--- Starting Training Loop from Epoch {start_epoch + 1} ---")
    # *** Load Scaler State AFTER loop starts and scaler is initialized ***
    if scaler_state_to_load and use_amp:
         try:
             scaler.load_state_dict(scaler_state_to_load)
             logger.info("Successfully loaded previous GradScaler state.")
         except Exception as scaler_load_e:
              logger.warning(f"Could not load GradScaler state: {scaler_load_e}. Starting with default scale.")
    elif use_amp:
         logger.info("Starting with default GradScaler state.")


    try:
        for epoch in range(start_epoch, epochs):
            current_epoch_num = epoch + 1
            logger.info(f"Starting Epoch {current_epoch_num}/{epochs}")

            # --- Epoch Training ---
            model.train()
            epoch_train_loss = 0.0
            batches_processed_in_epoch = 0
            optimizer.zero_grad(set_to_none=True) # Zero grad at epoch start

            progress_bar = tqdm(train_loader, desc=f"Epoch {current_epoch_num}", leave=False)
            for step, batch in enumerate(progress_bar):
                if batch is None: continue

                try:
                    # Move Batch to Device
                    pixel_values = batch['pixel_values'].to(device, non_blocking=pin_memory_flag)
                    labels = batch['labels'].to(device, non_blocking=pin_memory_flag)
                    label_lengths = batch['label_lengths'].to(device, non_blocking=pin_memory_flag)
                    current_batch_size = pixel_values.size(0)

                    # Autocast context manager
                    with autocast(enabled=use_amp):
                        # Forward pass
                        outputs = model(
                            pixel_values=pixel_values,
                            labels=labels,
                            label_lengths=label_lengths
                        )
                        loss = outputs.get('loss')

                        if loss is None or torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"Invalid loss ({loss}) detected at step {step}. Skipping.")
                            continue

                        # Apply overall weight if needed (usually 1.0)
                        loss = loss * ctc_loss_weight
                        normalized_loss = loss / grad_accumulation_steps

                    # --- Scaled Backward Pass ---
                    scaler.scale(normalized_loss).backward()

                    # --- Accumulate Loss for Logging ---
                    epoch_train_loss += loss.item()
                    batches_processed_in_epoch += 1

                    # --- Optimizer Step Logic ---
                    if (step + 1) % grad_accumulation_steps == 0:
                        # --- Defensive Scaler Handling ---
                        if use_amp:
                            try:
                                # Unscale BEFORE clipping
                                scaler.unscale_(optimizer) # Call unscale_
                            except RuntimeError as unscale_e:
                                # If unscale fails (e.g., already called), log and skip step
                                logger.error(f"scaler.unscale_() failed at step {step}, opt_step {optimizer_steps}: {unscale_e}", exc_info=False) # Less verbose log
                                logger.warning("Skipping optimizer step due to unscale error.")
                                # Need to zero grads manually if skipping step but backward happened
                                optimizer.zero_grad(set_to_none=True)
                                continue # Skip the rest of the optimizer step

                            # Clip gradients (AFTER unscaling)
                            try:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            except RuntimeError as clip_err:
                                logger.warning(f"Could not clip gradients at step {step+1}, opt_step {optimizer_steps}: {clip_err}. Skipping clipping.")

                            # Optimizer Step (wrapped by scaler)
                            scaler.step(optimizer) # Returns None if step was skipped

                            # Update scaler
                            scaler.update()

                        else: # Not using AMP
                            # Clip gradients directly
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            # Standard optimizer step
                            optimizer.step()

                        # Zero gradients AFTER step/update for the NEXT accumulation cycle
                        optimizer.zero_grad(set_to_none=True)

                        # LR Scheduler Step (AFTER optimizer step)
                        # Only step non-plateau schedulers here - plateau scheduler will be updated after validation
                        if lr_scheduler is not None and not is_plateau_scheduler:
                            lr_scheduler.step()
                        # --- End Defensive Scaler Handling ---

                        optimizer_steps += 1
                        # --- Logging & Optional Eval ---
                        current_lr = optimizer.param_groups[0]['lr']
                        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
                        if wandb_run and optimizer_steps % log_interval == 0:
                             wandb_run.log({
                                 "train/batch_ctc_loss": loss.item(),
                                 "train/learning_rate": current_lr,
                                 "train/step": optimizer_steps,
                                 "train/scaler_scale": scaler.get_scale() if use_amp else 1.0
                             })
                        if eval_steps is not None and optimizer_steps > 0 and optimizer_steps % eval_steps == 0:
                            logger.info(f"Running evaluation at step {optimizer_steps}...")
                            val_metrics = compute_ctc_validation_metrics(model, val_loader, device, validation_decoder)
                            model.train()
                            if wandb_run: wandb_run.log({f"val_step/{k}": v for k, v in val_metrics.items()}, step=optimizer_steps)

                            # For plateau scheduler, we need to pass the validation metric
                            if is_plateau_scheduler:
                                # Get the value of early_stopping_metric
                                plateau_metric_value = val_metrics.get(early_stopping_metric, float('inf'))
                                # Step the scheduler with this metric
                                logger.info(f"Stepping plateau scheduler with {early_stopping_metric} = {plateau_metric_value:.4f}")
                                lr_scheduler.step(plateau_metric_value)

                            current_metric_value = val_metrics.get(best_metric_name, float('inf') if not higher_is_better else -float('inf'))
                            is_best_step = (current_metric_value < best_metric_value) if not higher_is_better else (current_metric_value > best_metric_value)
                            if is_best_step:
                                logger.info(f"New best metric ({best_metric_name}) at step {optimizer_steps}: {current_metric_value:.4f}")
                                best_metric_value = current_metric_value; no_improvement_count = 0
                                best_model_state_dict = model.state_dict()
                            else: no_improvement_count += 1

                # --- Handle Batch Error ---
                except Exception as batch_e:
                    logger.error(f"Error processing batch {step} in epoch {current_epoch_num}: {batch_e}", exc_info=True)
                    optimizer.zero_grad(set_to_none=True) # Reset grads
                    if 'CUDA out of memory' in str(batch_e): logger.critical("CUDA Out of Memory.")
                    continue

            # --- End of Batch Loop ---

            # --- End of Epoch Validation, Logging, Saving ---
            avg_train_loss = epoch_train_loss / batches_processed_in_epoch if batches_processed_in_epoch > 0 else 0.0
            logger.info(f"Running end-of-epoch {current_epoch_num} validation...")
            val_metrics = compute_ctc_validation_metrics(model, val_loader, device, validation_decoder)
            model.train()
            epoch_val_metric = val_metrics.get(early_stopping_metric, float('inf') if not higher_is_better else -float('inf'))

            # Update plateau scheduler based on validation metric at end of epoch
            if is_plateau_scheduler:
                logger.info(f"Stepping plateau scheduler with {early_stopping_metric} = {epoch_val_metric:.4f}")
                lr_scheduler.step(epoch_val_metric)
            
            # Log Epoch Metrics
            if wandb_run: wandb_run.log({"epoch": current_epoch_num, "train/loss_epoch": avg_train_loss, **{f"val/{k}": v for k, v in val_metrics.items()}, "progress/no_improvement_count": no_improvement_count, "progress/learning_rate_epoch": optimizer.param_groups[0]['lr']})

            # Print Epoch Summary
            logger.info(f"--- Epoch {current_epoch_num}/{epochs} Summary ---"); logger.info(f"Avg Train Loss: {avg_train_loss:.4f}"); logger.info(f"Validation Metrics: {val_metrics}"); logger.info(f"Best Val Metric ({best_metric_name}) So Far: {best_metric_value:.4f}"); logger.info("---------------------")

            # Save Checkpoint
            is_best_epoch = (epoch_val_metric < best_metric_value) if not higher_is_better else (epoch_val_metric > best_metric_value)
            current_epoch_state = {'epoch': epoch, 'step': optimizer_steps, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None, 'scaler_state_dict': scaler.state_dict() if use_amp else None, 'best_val_metric': best_metric_value if not is_best_epoch else epoch_val_metric, 'current_val_metrics': val_metrics}
            checkpoint_path = os.path.join(checkpoints_dir, save_checkpoint_prefix)
            save_checkpoint(current_epoch_state, checkpoint_path, is_best=is_best_epoch)

            # Update best metric tracking
            if is_best_epoch:
                logger.info(f"New best metric ({best_metric_name}) at epoch {current_epoch_num}: {epoch_val_metric:.4f}")
                best_metric_value = epoch_val_metric; no_improvement_count = 0
                best_model_state_dict = model.state_dict()
                best_model_hf_path = os.path.join(output_dir, "best_model_hf")
                try: model.save_pretrained(best_model_hf_path)
                except Exception as e: logger.error(f"Could not save best HF model: {e}")
            else: no_improvement_count += 1
            logger.info(f"Early Stopping Check: no_improvement_count={no_improvement_count}, patience={early_stopping_patience}")

            # Early Stopping Check
            if no_improvement_count >= early_stopping_patience:
                logger.warning(f"Early stopping triggered.")
                break

            # End of Epoch Cleanup
            gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()
        # --- End of Training Loop (for epoch...) ---

    # --- Handle Exceptions/Finally block ---
    except KeyboardInterrupt: training_exception = "KeyboardInterrupt"; logger.warning("--- Training interrupted ---")
    except Exception as e: training_exception = e; logger.error(f"--- Training failed: {e} ---", exc_info=True)
    finally:
        final_epoch = epoch if 'epoch' in locals() else start_epoch - 1
        logger.info(f"Training loop ended. Last epoch: {final_epoch}. Steps: {optimizer_steps}")
        logger.info("Attempting to save final state...")
        try:
            final_state = {'epoch': final_epoch,'step': optimizer_steps, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None, 'scaler_state_dict': scaler.state_dict() if use_amp else None, 'best_val_metric': best_metric_value}
            final_chkpt_path = os.path.join(checkpoints_dir, save_checkpoint_prefix)
            save_checkpoint(final_state, final_chkpt_path, is_best=False)
        except Exception as final_save_e: logger.error(f"Could not save final checkpoint: {final_save_e}")

        if not training_exception:
            logger.info("Training completed normally.")
            if best_model_state_dict is not None:
                 logger.info("Restoring best model weights...")
                 try: model.load_state_dict(best_model_state_dict)
                 except Exception as load_e: logger.error(f"Load best failed: {load_e}")
            final_model_hf_path = os.path.join(output_dir, "final_model_hf")
            try: model.save_pretrained(final_model_hf_path); logger.info(f"Saved final model (best weights) to {final_model_hf_path}")
            except Exception as save_e: logger.error(f"Save final failed: {save_e}")

        if wandb_run:
            exit_code = 1 if training_exception else 0; logger.info(f"Finishing wandb run (code: {exit_code})...")
            try: wandb_run.finish(exit_code=exit_code)
            except Exception as wandb_e: logger.error(f"Wandb finish error: {wandb_e}")

        if training_exception and not isinstance(training_exception, KeyboardInterrupt): raise training_exception

    return model