# scripts/train_hierarchical_ctc.py
import os
import sys
import argparse
import torch
from datasets import load_dataset, DatasetDict
import logging
import math
import wandb  # Optional
import json
import numpy as np

# --- Adjust imports ---
from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel, HierarchicalCtcOcrConfig
from data.ctc_ocr_dataset import CtcOcrDataset  # Reuse standard CTC dataset
from data.ctc_collation import ctc_collate_fn  # Reuse standard CTC collate
from training.ctc_trainer import train_ctc_model  # Reuse standard CTC trainer
from utils.schedulers import CosineWarmupScheduler, CosineWarmupWithPlateauScheduler
from utils.optimizers import create_optimizer
from utils.ctc_utils import build_ctc_vocab, build_combined_vietnamese_charset  # Import builders

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('main_training_hierarchical.log'),
    ],
)
logger = logging.getLogger('TrainHierarchicalCtcMultiScaleScript')

# --- Define Base/Diacritic Vocabs (Needed for model config, even if not used in loss) ---
# These should ideally match the ones used in dual_ctc for consistency if comparing
BASE_CHAR_VOCAB_HIER = [
    '<blank>',
    '<unk>',
    'a',
    'b',
    'c',
    'd',
    'e',
    'g',
    'h',
    'i',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'v',
    'x',
    'y',
    'A',
    'B',
    'C',
    'D',
    'E',
    'G',
    'H',
    'I',
    'K',
    'L',
    'M',
    'N',
    'O',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'U',
    'V',
    'X',
    'Y',
    'đ',
    'Đ',
    'f',
    'F',
    'j',
    'J',
    'w',
    'W',
    'z',
    'Z',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    ' ',
    ',',
    '.',
    '?',
    '!',
    ':',
    ';',
    '-',
    '_',
    '(',
    ')',
    '[',
    ']',
    '{',
    '}',
    "'",
    '"',
    '/',
    '\\',
    '@',
    '#',
    '$',
    '%',
    '^',
    '&',
    '*',
    '+',
    '=',
    '<',
    '>',
    '|',
]
DIACRITIC_VOCAB_HIER = [
    '<blank>',
    'no_diacritic',
    '<unk>',
    'acute',
    'grave',
    'hook',
    'tilde',
    'dot',
    'circumflex',
    'breve',
    'horn',
    'stroke',
    'circumflex_grave',
    'circumflex_acute',
    'circumflex_tilde',
    'circumflex_hook',
    'circumflex_dot',
    'breve_grave',
    'breve_acute',
    'breve_tilde',
    'breve_hook',
    'breve_dot',
    'horn_grave',
    'horn_acute',
    'horn_tilde',
    'horn_hook',
    'horn_dot',
]


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical CTC Vietnamese OCR model')

    # --- Arguments ---
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='vklinhhh/vietnamese_handwriting_ocr',
        help='HF dataset (image, label)',
    )  # Need combined label
    parser.add_argument(
        '--vision_encoder',
        type=str,
        default='microsoft/trocr-base-handwritten',
        help='Vision encoder',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/hierarchical_ctc_ocr_model',
        help='Output directory',
    )
    parser.add_argument(
        '--combined_char_vocab_json',
        type=str,
        default=None,
        help='Path to JSON list of FINAL combined characters (incl. blank/unk). If None, uses default generator.',
    )

    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--load_weights_from', type=str, default=None)
    parser.add_argument('--fusion_layers', type=str, default="-1,-4", help='Comma-separated list of negative/positive vision encoder layer indices to fuse (e.g., "-1,-4").')
    parser.add_argument('--fusion_method', type=str, default="concat_proj", choices=['concat_proj', 'add', 'bilinear', 'none'], help='Method to fuse features from different layers.')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=7e-8)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument(
        '--early_stopping_metric', type=str, default='val_cer'
    )  # Monitor CER on combined chars

    # Model Architecture options
    parser.add_argument('--rnn_hidden_size', type=int, default=512)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--rnn_dropout', type=float, default=0.1)
    parser.add_argument('--no_bidirectional', action='store_true')
    parser.add_argument(
        '--shared_hidden_size',
        type=int,
        default=512,
        help='Hidden size after RNN before branching.',
    )
    parser.add_argument(
        '--conditioning_method',
        type=str,
        default='concat_proj',
        choices=['concat_proj', 'gate', 'none'],
        help='Method to condition diacritic head.',
    )
    parser.add_argument('--classifier_dropout', type=float, default=0.2)

    # Logging, System params
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=5000)
    parser.add_argument('--eval_steps', type=int, default=None)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--discriminative_lr', action='store_true')
    parser.add_argument('--skip_final_eval', action='store_true')
    parser.add_argument('--num_shared_layers', type=int, default=1)
    parser.add_argument('--encoder_lr_factor', type=float, default=0.1)
    parser.add_argument('--reset_scheduler_on_resume', action='store_true',
                        help="Reinitialize the LR scheduler when resuming training (useful if extending total epochs).")
    
    # NEW: Enhanced Diacritic Classifier Arguments
    parser.add_argument('--use_visual_diacritic_attention', action='store_true',
                       help='Enable Visual Diacritic Attention mechanism')
    parser.add_argument('--use_character_diacritic_compatibility', action='store_true',
                       help='Enable explicit Character-Diacritic Compatibility matrix')
    parser.add_argument('--use_few_shot_diacritic_adapter', action='store_true',
                       help='Enable Few-Shot Diacritic Adapter for rare combinations')
    parser.add_argument('--num_few_shot_prototypes', type=int, default=5,
                       help='Number of prototypes per diacritic class for few-shot adapter')
    # NEW: LR Scheduler Arguments
    parser.add_argument('--use_reduce_lr_on_plateau', action='store_true',
                       help='Use ReduceLROnPlateau scheduler based on validation CER')
    parser.add_argument('--plateau_patience', type=int, default=3,
                       help='Number of epochs with no improvement after which LR will be reduced')
    parser.add_argument('--plateau_factor', type=float, default=0.5,
                       help='Factor by which the learning rate will be reduced')
    parser.add_argument('--plateau_min_lr', type=float, default=1e-7,
                       help='Minimum learning rate for plateau scheduler')
    parser.add_argument(
    '--no_augment',
    action='store_true', # This makes it a flag, default is False
    help="Disable training data augmentations."
)
    args = parser.parse_args()

    # --- Setup ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    run_name_to_use = args.wandb_run_name or os.path.basename(args.output_dir) or "hier_ctc_multiscale_run" 
    # --- Parse Fusion Layers ---
    try:
        fusion_layer_indices = [int(x.strip()) for x in args.fusion_layers.split(',')]
        logger.info(f"Using fusion layers: {fusion_layer_indices}")
    except:
        logger.error(f"Invalid format for --fusion_layers '{args.fusion_layers}'. Using default [-1, -4].")
        fusion_layer_indices = [-1, -4]
    # --- Build COMBINED Character Vocabulary for final CTC output ---
    if args.combined_char_vocab_json and os.path.exists(args.combined_char_vocab_json):
        logger.info(f'Loading COMBINED char vocab from: {args.combined_char_vocab_json}')
        with open(args.combined_char_vocab_json, 'r', encoding='utf-8') as f:
            combined_char_list = json.load(f)
        # Ensure blank/unk
        if '<blank>' not in combined_char_list or combined_char_list[0] != '<blank>':
            if '<blank>' in combined_char_list:
                combined_char_list.remove('<blank>')
            combined_char_list.insert(0, '<blank>')
        if '[UNK]' not in combined_char_list:
            combined_char_list.append('[UNK]')
        combined_vocab, combined_char_to_idx, combined_idx_to_char = build_ctc_vocab(
            combined_char_list, add_blank=False, add_unk=False
        )
    else:
        logger.info('Building COMBINED char vocab from generator...')
        generated_combined_chars = build_combined_vietnamese_charset()
        combined_vocab, combined_char_to_idx, combined_idx_to_char = build_ctc_vocab(
            generated_combined_chars, add_blank=True, add_unk=True, unk_token='[UNK]'
        )
        # Save the generated vocab
        os.makedirs(args.output_dir, exist_ok=True)
        vocab_save_path = os.path.join(args.output_dir, 'combined_char_vocab.json')
        with open(vocab_save_path, 'w', encoding='utf-8') as f:
            json.dump(combined_vocab, f, ensure_ascii=False, indent=4)
        logger.info(f'Saved generated COMBINED vocab to {vocab_save_path}')

    # --- Load Dataset ---
    try:
        logger.info(f'Loading dataset: {args.dataset_name}')
        hf_dataset = load_dataset(args.dataset_name)
        if 'validation' not in hf_dataset or 'train' not in hf_dataset:
            logger.warning(f'Splitting train set for validation.')
            if args.val_split <= 0:
                raise ValueError('--val_split required')
            split_dataset = hf_dataset['train'].train_test_split(
                test_size=args.val_split, seed=args.seed
            )
            hf_dataset = DatasetDict(
                {'train': split_dataset['train'], 'validation': split_dataset['test']}
            )
        train_hf_split = hf_dataset['train']
        val_hf_split = hf_dataset['validation']
        logger.info(f'Train size: {len(train_hf_split)}, Val size: {len(val_hf_split)}')
    except Exception as dataset_load_e:
        logger.error(f'FATAL: Dataset load failed: {dataset_load_e}', exc_info=True)
        return 1

    # --- Initialize Model ---
    try:
        logger.info("Initializing HierarchicalCtcMultiScaleOcrModel configuration...")
        model_config = HierarchicalCtcOcrConfig( # Use the same config class name
            vision_encoder_name=args.vision_encoder,
            base_char_vocab=BASE_CHAR_VOCAB_HIER,
            diacritic_vocab=DIACRITIC_VOCAB_HIER,
            combined_char_vocab=combined_vocab,
            # *** Pass fusion args ***
            vision_encoder_layer_indices=fusion_layer_indices,
            feature_fusion_method=args.fusion_method,
            # *** End fusion args ***
            intermediate_rnn_layers=args.rnn_layers,
            rnn_hidden_size=args.rnn_hidden_size,
            rnn_dropout=args.rnn_dropout,
            rnn_bidirectional=not args.no_bidirectional,
            shared_hidden_size=args.shared_hidden_size,
            conditioning_method=args.conditioning_method,
            classifier_dropout=args.classifier_dropout,
            blank_idx=combined_char_to_idx['<blank>'],
            num_shared_layers=args.num_shared_layers,
            # NEW: Enhanced Diacritic Classifier settings
            use_visual_diacritic_attention=args.use_visual_diacritic_attention,
            use_character_diacritic_compatibility=args.use_character_diacritic_compatibility,
            use_few_shot_diacritic_adapter=args.use_few_shot_diacritic_adapter,
            num_few_shot_prototypes=args.num_few_shot_prototypes,
        )

        model_load_path = args.load_weights_from if args.load_weights_from else args.vision_encoder
        logger.info(f"Instantiating/Loading model from: {model_load_path}")
        init_kwargs = model_config.to_dict()
        model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(model_load_path, **init_kwargs) # Use NEW model class name
        processor = model.processor
        logger.info("Model and Processor initialized.")
        
        # Log information about enhanced diacritic classifiers
        if args.use_visual_diacritic_attention:
            logger.info("Visual Diacritic Attention mechanism ENABLED")
        if args.use_character_diacritic_compatibility:
            logger.info("Character-Diacritic Compatibility matrix ENABLED")
        if args.use_few_shot_diacritic_adapter:
            logger.info(f"Few-Shot Diacritic Adapter ENABLED with {args.num_few_shot_prototypes} prototypes per class")

    except Exception as model_init_e: logger.error(f"FATAL: Model init failed: {model_init_e}", exc_info=True); return 1

    # --- Create Datasets ---
    try:
        logger.info('Creating CTC dataset wrappers (using combined vocab)...')
        # Use the standard CtcOcrDataset, but pass the COMBINED char map
        train_dataset = CtcOcrDataset(
            train_hf_split, processor, combined_char_to_idx, unk_token='[UNK]', is_training=not args.no_augment
        )
        val_dataset = CtcOcrDataset(
            val_hf_split, processor, combined_char_to_idx, unk_token='[UNK]', is_training=False
        )
        logger.info('CTC Dataset wrappers created.')
    except Exception as dataset_wrap_e:
        logger.error(f'FATAL: Dataset wrap failed: {dataset_wrap_e}', exc_info=True)
        return 1
    # --- Move model to device ---
    model.to(device)
    logger.info(f'Model on device: {device}')
    try:
        optimizer = create_optimizer(
            model,  
            args.learning_rate,
            args.weight_decay,
            args.discriminative_lr,
            args.encoder_lr_factor,
        )
        num_training_batches = math.ceil(len(train_dataset) / args.batch_size)
        total_steps = math.ceil(num_training_batches / args.grad_accumulation) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        logger.info(f'Scheduler Setup: Steps={total_steps}, Warmup={warmup_steps}')
        # NEW: Create scheduler based on configuration
        if args.use_reduce_lr_on_plateau:
            # Create combined scheduler with plateau capability
            logger.info(f"Using CosineWarmupWithPlateauScheduler: "
                       f"plateau_patience={args.plateau_patience}, "
                       f"plateau_factor={args.plateau_factor}, "
                       f"metric='val_cer'")
            lr_scheduler = CosineWarmupWithPlateauScheduler(
                optimizer=optimizer,
                warmup_steps=warmup_steps,
                max_steps=int(total_steps * 0.5),  # Use plateau for latter half
                plateau_patience=args.plateau_patience,
                plateau_factor=args.plateau_factor,
                plateau_min_lr=args.plateau_min_lr,
                plateau_metric='val_cer',  # Hardcoded to monitor CER
                plateau_mode='min'  # CER is better when lower
            )
        else:
            # Use standard cosine warmup scheduler
            logger.info(f"Using standard CosineWarmupScheduler")
            lr_scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
        
        logger.info('Optimizer/Scheduler created.')
    except Exception as opt_sched_e:
        logger.error(f'FATAL: Opt/Sched failed: {opt_sched_e}', exc_info=True)
        return 1

    # --- Handle Resuming (Full State) ---
    start_epoch = 0; resumed_optimizer_steps = 0
    higher_is_better = args.early_stopping_metric not in ['val_loss', 'val_cer', 'val_wer'] # Hierarchical only uses combined CER/WER
    resumed_best_val_metric = -float('inf') if higher_is_better else float('inf')
    scaler_state_to_load = None # Initialize scaler state holder
    checkpoint_to_load = args.resume_from_checkpoint

    if checkpoint_to_load is None and not args.load_weights_from:
        latest_checkpoint_path = os.path.join(args.output_dir, "checkpoints", "checkpoint_latest.pt")
        checkpoint_to_load = latest_checkpoint_path if os.path.isfile(latest_checkpoint_path) else None
        if checkpoint_to_load: logger.info(f"Found latest ckpt: {checkpoint_to_load}")
        else: logger.info("No checkpoint found.")


    if checkpoint_to_load and os.path.isfile(checkpoint_to_load):
        logger.info(f"--- Loading checkpoint state: {checkpoint_to_load} ---")
        try:
            checkpoint = torch.load(checkpoint_to_load, map_location=device)
            logger.info(f"Keys: {list(checkpoint.keys())}")

            if not args.load_weights_from and 'model_state_dict' in checkpoint:
                load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info(f"Model state loaded. Miss:{load_result.missing_keys}, Unexp:{load_result.unexpected_keys}")

            load_optimizer_etc = False
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer loaded.")
                load_optimizer_etc = True
            else: logger.warning("Optimizer state missing.")

            if load_optimizer_etc and lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                if args.reset_scheduler_on_resume:
                    logger.warning("Resetting LR scheduler state due to --reset_scheduler_on_resume flag.")    
                    num_training_batches_new = math.ceil(len(train_dataset) / args.batch_size)
                    remaining_epochs = args.epochs - start_epoch
                    total_steps_new = math.ceil(num_training_batches_new / args.grad_accumulation) * remaining_epochs
                    warmup_steps_new = int(total_steps_new * args.warmup_ratio)
                    # Create a NEW scheduler instance
                    optimizer = create_optimizer(
                                model,  
                                args.learning_rate,
                                args.weight_decay,
                                args.discriminative_lr,
                                args.encoder_lr_factor,
                            )
                    # Recreate scheduler based on configuration
                    if args.use_reduce_lr_on_plateau:
                        lr_scheduler = CosineWarmupWithPlateauScheduler(
                            optimizer=optimizer,
                            warmup_steps=warmup_steps_new,
                            max_steps=int(total_steps_new * 0.5),
                            plateau_patience=args.plateau_patience,
                            plateau_factor=args.plateau_factor,
                            plateau_min_lr=args.plateau_min_lr,
                            plateau_metric='val_cer'
                        )
                    else:
                        lr_scheduler = CosineWarmupScheduler(
                            optimizer, warmup_steps_new, total_steps_new, last_epoch=start_epoch-1
                        )
                    logger.info(f"Reinitialized LR scheduler for {remaining_epochs} remaining epochs.")
                else:
                    try:
                        if args.use_reduce_lr_on_plateau and isinstance(lr_scheduler, CosineWarmupWithPlateauScheduler):
                            # Special handling for combined scheduler
                            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                        else:
                            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                        logger.info("Scheduler state loaded from checkpoint.")
                    except Exception as sched_e:
                        logger.warning(f"Error loading scheduler state: {sched_e}. Creating fresh scheduler.")
                        # Fall back to recreating scheduler if loading fails
                        if args.use_reduce_lr_on_plateau:
                            lr_scheduler = CosineWarmupWithPlateauScheduler(
                                optimizer=optimizer,
                                warmup_steps=warmup_steps,
                                max_steps=int(total_steps * 0.5),
                                plateau_patience=args.plateau_patience,
                                plateau_factor=args.plateau_factor,
                                plateau_min_lr=args.plateau_min_lr,
                                plateau_metric='val_cer'
                            )
                        else:
                            lr_scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
            # *** ALWAYS try to load scaler state if found ***
            if load_optimizer_etc and args.use_amp and 'scaler_state_dict' in checkpoint:
                scaler_state_to_load = checkpoint['scaler_state_dict'] # Assign state
                logger.info("Found AMP state in checkpoint. Will load in trainer.")
            elif load_optimizer_etc and args.use_amp:
                logger.warning("Scaler state not found in checkpoint.")
            # *** END CHANGE ***

            start_epoch = checkpoint.get('epoch', -1) + 1
            resumed_optimizer_steps = checkpoint.get('step', 0)
            resumed_best_val_metric = checkpoint.get('best_val_metric', resumed_best_val_metric)
            logger.info(f"-> Resuming Epoch: {start_epoch} (Step: {resumed_optimizer_steps}), Best Metric: {resumed_best_val_metric:.4f}")

        except Exception as e:
            logger.error(f"ERROR loading checkpoint state: {e}", exc_info=True)
            start_epoch = 0; resumed_optimizer_steps = 0; resumed_best_val_metric = -float('inf') if higher_is_better else float('inf')
    else:
        # ... (Starting fresh message) ...
        if args.resume_from_checkpoint: logger.warning(f"Specified checkpoint not found: {args.resume_from_checkpoint}.")
        logger.info("Starting fresh or from base weights.")


    # --- WandB Init AFTER potential config updates ---
    wandb_run = None
    if wandb_run is None and args.wandb_project:  # Initialize if not already done
        try:
            wandb_config = vars(args).copy()
            wandb_config['base_char_vocab_size'] = len(BASE_CHAR_VOCAB_HIER)
            wandb_config['diacritic_vocab_size'] = len(DIACRITIC_VOCAB_HIER)
            wandb_config['combined_char_vocab_size'] = len(combined_vocab)
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name_to_use,
                config=wandb_config,
                resume='allow',
            )
            logger.info(f'Initialized WandB run: {wandb_run.name} (ID: {wandb_run.id})')
        except Exception as e:
            logger.error(f'Wandb init failed: {e}')

    # --- Start Training (Use the standard CTC trainer) ---
    logger.info('============ Starting Hierarchical CTC Training Phase ============')
    trained_model = train_ctc_model(  # Use the same trainer as single CTC
        model=model,  # Pass the hierarchical model
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataset=train_dataset,  # Uses combined vocab targets
        val_dataset=val_dataset,  # Uses combined vocab targets
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output_dir,
        start_epoch=start_epoch,
        resumed_optimizer_steps=resumed_optimizer_steps,
        resumed_best_val_metric=resumed_best_val_metric,
        best_metric_name=args.early_stopping_metric,
        project_name=args.wandb_project,  # Pass None if already init
        run_name=run_name_to_use,  # Pass None if already init
        log_interval=args.log_interval,
        save_checkpoint_prefix='checkpoint',
        use_amp=args.use_amp,
        scaler_state_to_load=scaler_state_to_load,  # Pass the scaler state if available
        grad_accumulation_steps=args.grad_accumulation,
        num_workers=args.num_workers,
        eval_steps=args.eval_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
    )

    logger.info(f'============ Hierarchical CTC Training finished ============')
    logger.info(f'Final model artifacts saved in {args.output_dir}')

    # --- Final Evaluation ( Reuse evaluate_ctc.py script ) ---
    if not args.skip_final_eval:
        logger.info('============ Starting Final Evaluation on Test Set ============')
        best_model_path = os.path.join(args.output_dir, 'best_model_hf')
        if not os.path.isdir(best_model_path):
            best_model_path = args.output_dir
        eval_output_dir = os.path.join(args.output_dir, 'final_evaluation_report')
        # Need to import the evaluate_ctc script's main function
        try:
            from scripts.evaluate_ctc import run_evaluation as run_ctc_evaluation  # Rename import

            run_ctc_evaluation(  # Call the standard CTC evaluation
                model_path=best_model_path,  # Path to hierarchical model
                dataset_name=args.test_dataset_name,
                output_dir=eval_output_dir,
                # Evaluation needs the COMBINED vocab path
                char_vocab_json=os.path.join(args.output_dir, 'combined_char_vocab.json'),
                dataset_split=args.test_dataset_split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
        except ImportError:
            logger.error('Could not import evaluate_ctc script.')
        except Exception as eval_e:
            logger.error(f'Final evaluation failed: {eval_e}', exc_info=True)

    return 0


if __name__ == '__main__':
    status = main()
    sys.exit(status)