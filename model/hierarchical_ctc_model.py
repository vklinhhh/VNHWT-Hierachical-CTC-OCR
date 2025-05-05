# model/hierarchical_ctc_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    VisionEncoderDecoderModel,
    PreTrainedModel,
    PretrainedConfig
)
import os
import json
import logging
import inspect # For checking kwargs in from_pretrained fallback

logger = logging.getLogger(__name__)

# --- Configuration Class ---
class HierarchicalCtcOcrConfig(PretrainedConfig):
    model_type = "hierarchical_ctc_ocr"

    def __init__(
        self,
        vision_encoder_name='microsoft/trocr-base-handwritten',
        # Vocabs needed for head sizes and potentially conditioning
        base_char_vocab=None, # For size of base head
        diacritic_vocab=None, # For size of potential diacritic head / conditioning
        combined_char_vocab=None, # Combined chars 'a','á','à' for FINAL CTC output
        # Intermediate layers
        intermediate_rnn_layers=2,
        rnn_hidden_size=512,
        rnn_dropout=0.1,
        rnn_bidirectional=True,
        # Shared / Hierarchical Layers
        shared_hidden_size=512, # Dimension after RNN, before branching
        conditioning_method="concat", # 'concat', 'gate', 'none'
        # Classifier dropout
        classifier_dropout=0.1,
        blank_idx=0, # For final combined vocab
        # Vision encoder config can be stored if needed (optional)
        vision_encoder_config=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_encoder_name = vision_encoder_name
        # Store lists directly in config
        self.base_char_vocab = base_char_vocab if base_char_vocab else []
        self.diacritic_vocab = diacritic_vocab if diacritic_vocab else []
        self.combined_char_vocab = combined_char_vocab if combined_char_vocab else []

        # Calculate sizes from lists
        self.base_char_vocab_size = len(self.base_char_vocab)
        self.diacritic_vocab_size = len(self.diacritic_vocab)
        self.combined_char_vocab_size = len(self.combined_char_vocab) # Final output size

        self.intermediate_rnn_layers = intermediate_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_dropout = rnn_dropout
        self.rnn_bidirectional = rnn_bidirectional
        self.shared_hidden_size = shared_hidden_size
        self.conditioning_method = conditioning_method
        self.classifier_dropout = classifier_dropout
        self.blank_idx = blank_idx
        self.vision_encoder_config = vision_encoder_config # Store this too

        if not self.combined_char_vocab:
            # Allow init without combined if loading base, but log warning
            logger.warning("Combined character vocabulary is empty during config init.")
        # No error needed here, check happens in model __init__


# --- Model Class ---
class HierarchicalCtcOcrModel(PreTrainedModel):
    config_class = HierarchicalCtcOcrConfig

    def __init__(self, config: HierarchicalCtcOcrConfig):
        super().__init__(config)
        self.config = config # Store config on the model instance

        # *** Ensure vocabs needed for layer sizes are present ***
        if not config.combined_char_vocab:
            raise ValueError("Combined character vocabulary must be provided in the config during model initialization.")
        if config.conditioning_method != 'none' and (not config.base_char_vocab or not config.diacritic_vocab):
             raise ValueError("Base/Diacritic vocabs must be provided in config for conditioning methods 'concat' or 'gate'.")

        # Store vocabs directly on model instance for easy access later
        self.base_char_vocab = config.base_char_vocab
        self.diacritic_vocab = config.diacritic_vocab
        self.combined_char_vocab = config.combined_char_vocab

        logger.info("Initializing HierarchicalCtcOcrModel...")

        # --- Load Processor & Vision Encoder ---
        try:
            self.processor = AutoProcessor.from_pretrained(config.vision_encoder_name)
            base_model = VisionEncoderDecoderModel.from_pretrained(config.vision_encoder_name)
            self.vision_encoder = base_model.encoder
            # Update config with actual loaded encoder config if not passed explicitly
            if self.config.vision_encoder_config is None:
                 self.config.vision_encoder_config = self.vision_encoder.config
            logger.info("Processor and Vision Encoder loaded.")
            del base_model
        except Exception as e:
            logger.error(f"Failed loading base components: {e}", exc_info=True); raise

        # --- Intermediate Layers ---
        encoder_output_size = self.vision_encoder.config.hidden_size
        self.rnn = nn.GRU(
            input_size=encoder_output_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.intermediate_rnn_layers,
            batch_first=True,
            dropout=config.rnn_dropout if config.intermediate_rnn_layers > 1 else 0,
            bidirectional=config.rnn_bidirectional
        )
        rnn_output_size = config.rnn_hidden_size * 2 if config.rnn_bidirectional else config.rnn_hidden_size

        self.shared_layer = nn.Sequential(
             nn.Linear(rnn_output_size, config.shared_hidden_size),
             nn.LayerNorm(config.shared_hidden_size),
             nn.GELU(),
             nn.Dropout(config.classifier_dropout)
        )
        logger.info(f"Added shared feature layer (Output: {config.shared_hidden_size})")

        # --- Hierarchical Branching ---
        self.base_classifier = nn.Linear(config.shared_hidden_size, config.base_char_vocab_size)
        logger.info(f"Added Base Classifier Head (Output: {config.base_char_vocab_size})")

        conditioning_input_size = config.shared_hidden_size
        self.diacritic_gate = None # Initialize gate to None
        if config.conditioning_method == 'concat':
            conditioning_input_size += config.base_char_vocab_size # Shared + Base Logits
            logger.info("Using 'concat' conditioning for diacritic head.")
        elif config.conditioning_method == 'gate':
            self.diacritic_gate = nn.Sequential(
                nn.Linear(config.shared_hidden_size + config.base_char_vocab_size, config.shared_hidden_size),
                nn.Sigmoid()
            )
            logger.info("Using 'gate' conditioning for diacritic head.")
            # Input size remains shared_hidden_size, applied via multiplication

        self.diacritic_classifier = nn.Linear(conditioning_input_size, config.diacritic_vocab_size)
        logger.info(f"Added Diacritic Classifier Head (Input: {conditioning_input_size}, Output: {config.diacritic_vocab_size})")

        # --- Final Combined Classifier ---
        final_combiner_input_size = config.shared_hidden_size
        self.final_classifier = nn.Linear(final_combiner_input_size, config.combined_char_vocab_size)
        logger.info(f"Added Final Combined Classifier Head (Input: {final_combiner_input_size}, Output: {config.combined_char_vocab_size})")

        # Initialize weights for new layers
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for RNN, Shared Layer, and Classifier Heads."""
        logger.debug("Initializing new layer weights...")
        for name, param in self.rnn.named_parameters():
            if 'bias' in name: nn.init.zeros_(param)
            elif 'weight' in name: nn.init.xavier_uniform_(param)
        for layer in self.shared_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None: nn.init.zeros_(layer.bias)
        for head in [self.base_classifier, self.diacritic_classifier, self.final_classifier]:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None: nn.init.zeros_(head.bias)
        # Init gate layers if they exist
        if self.diacritic_gate:
             for layer in self.diacritic_gate:
                  if isinstance(layer, nn.Linear):
                      nn.init.xavier_uniform_(layer.weight)
                      if layer.bias is not None: nn.init.zeros_(layer.bias)


    def forward(
        self,
        pixel_values,
        labels=None,       # Padded COMBINED char indices [B, MaxLabelLen]
        label_lengths=None # Actual lengths of label sequences [B]
        ):
        # 1. Vision Encoder
        encoder_outputs = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        hidden_states = encoder_outputs.last_hidden_state

        # 2. RNN Layers
        rnn_outputs, _ = self.rnn(hidden_states)

        # 3. Shared Feature Layer
        shared_features = self.shared_layer(rnn_outputs) # [B, T, D_shared]

        # 4. Base Character Branch
        base_logits = self.base_classifier(shared_features) # [B, T, N_base]

        # 5. Diacritic Character Branch (with Conditioning)
        diacritic_input_features = shared_features
        if self.config.conditioning_method == 'concat':
            diacritic_input_features = torch.cat((shared_features, base_logits), dim=-1)
        elif self.config.conditioning_method == 'gate' and self.diacritic_gate is not None:
            gate_input = torch.cat((shared_features, base_logits), dim=-1)
            diacritic_gate_values = self.diacritic_gate(gate_input)
            diacritic_input_features = shared_features * diacritic_gate_values # Gated features

        diacritic_logits = self.diacritic_classifier(diacritic_input_features) # [B, T, N_diac]

        # 6. Final Combined Classifier
        final_input_features = shared_features # Using simplest option
        final_logits = self.final_classifier(final_input_features) # [B, T, N_combined]

        # --- Prepare for CTC Loss ---
        log_probs = final_logits.log_softmax(dim=2).permute(1, 0, 2) # [T, B, N_combined]
        time_steps = log_probs.size(0)

        # --- Calculate Final CTC Loss ---
        loss = None
        if labels is not None and label_lengths is not None:
            batch_size_actual = log_probs.size(1)
            input_lengths = torch.full((batch_size_actual,), time_steps, dtype=torch.long, device='cpu')
            labels_cpu = labels.cpu()
            label_lengths_cpu = label_lengths.cpu()
            input_lengths_clamped = torch.clamp(input_lengths, max=time_steps)
            label_lengths_clamped = torch.clamp(label_lengths_cpu, max=labels_cpu.size(1))
            try:
                ctc_loss_fn = nn.CTCLoss(blank=self.config.blank_idx, reduction='mean', zero_infinity=True)
                loss = ctc_loss_fn(log_probs, labels_cpu, input_lengths_clamped, label_lengths_clamped)
            except Exception as e:
                logger.error(f"Error calculating FINAL CTC loss: {e}", exc_info=True)
                loss = torch.tensor(0.0, device=final_logits.device, requires_grad=True)

        return {
            'loss': loss, # Final combined CTC loss
            'logits': final_logits, # Final combined logits [B, T, N_combined]
            'base_logits': base_logits, # Intermediate logits
            'diacritic_logits': diacritic_logits, # Intermediate logits
        }

    # --- Save/Load Methods ---
    def save_pretrained(self, save_directory, **kwargs):
        """Saves the model configuration and state dictionary."""
        logger.info(f"Saving {self.__class__.__name__} model to: {save_directory}")
        os.makedirs(save_directory, exist_ok=True)

        # *** Ensure vocabs stored on the instance are in the config object ***
        if hasattr(self, 'base_char_vocab'): self.config.base_char_vocab = self.base_char_vocab
        if hasattr(self, 'diacritic_vocab'): self.config.diacritic_vocab = self.diacritic_vocab
        if hasattr(self, 'combined_char_vocab'): self.config.combined_char_vocab = self.combined_char_vocab
        # Update sizes in config based on instance attributes
        self.config.base_char_vocab_size = len(getattr(self, 'base_char_vocab', []))
        self.config.diacritic_vocab_size = len(getattr(self, 'diacritic_vocab', []))
        self.config.combined_char_vocab_size = len(getattr(self, 'combined_char_vocab', []))

        # Save the config
        self.config.save_pretrained(save_directory)

        # Save the full state dict
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), output_model_file)

        # Save processor
        try:
            if hasattr(self, 'processor') and self.processor:
                 self.processor.save_pretrained(save_directory)
            else:
                 logger.warning("Processor not found on model instance, not saving processor.")
        except Exception as e:
             logger.error(f"Failed to save processor: {e}")

        logger.info(f"Model components saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, **kwargs):
        """Loads the model configuration and state dictionary."""
        logger.info(f"Loading {cls.__name__} from: {pretrained_model_name_or_path}")
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        loaded_config = None

        # 1. Determine Config (Prioritize passed config -> file -> kwargs)
        if config is not None and isinstance(config, cls.config_class):
            logger.info("Using provided config object.")
            loaded_config = config
        elif os.path.exists(config_path):
            logger.info(f"Loading config from file: {config_path}")
            # Load config using its own from_pretrained method
            loaded_config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
            # Override vocabs from kwargs if they were explicitly passed
            if 'base_char_vocab' in kwargs and kwargs['base_char_vocab']: loaded_config.base_char_vocab = kwargs['base_char_vocab']; loaded_config.base_char_vocab_size = len(kwargs['base_char_vocab'])
            if 'diacritic_vocab' in kwargs and kwargs['diacritic_vocab']: loaded_config.diacritic_vocab = kwargs['diacritic_vocab']; loaded_config.diacritic_vocab_size = len(kwargs['diacritic_vocab'])
            if 'combined_char_vocab' in kwargs and kwargs['combined_char_vocab']: loaded_config.combined_char_vocab = kwargs['combined_char_vocab']; loaded_config.combined_char_vocab_size = len(kwargs['combined_char_vocab'])
        else:
            logger.warning(f"Config file not found at {config_path}. Initializing new config from kwargs.")
            # Need all vocabs in kwargs if creating config from scratch
            if not all(k in kwargs for k in ['base_char_vocab', 'diacritic_vocab', 'combined_char_vocab']):
                 raise ValueError("All vocabs (base, diacritic, combined) must be in kwargs when initializing without config.json.")
            # Set vision encoder name if missing in kwargs
            if 'vision_encoder_name' not in kwargs and hasattr(cls.config_class, '__init__'):
                 sig = inspect.signature(cls.config_class.__init__)
                 if 'vision_encoder_name' in sig.parameters: kwargs['vision_encoder_name'] = sig.parameters['vision_encoder_name'].default
            loaded_config = cls.config_class(**kwargs)

        # 2. Instantiate Model using the determined config
        model = cls(loaded_config)

        # 3. Load State Dictionary
        state_dict_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        is_loading_base_model = not os.path.exists(config_path) # True if we loaded from e.g., 'microsoft/trocr...'

        if os.path.exists(state_dict_path) and not is_loading_base_model:
            logger.info(f"Loading state dict from: {state_dict_path}")
            try:
                 state_dict = torch.load(state_dict_path, map_location="cpu")
                 load_result = model.load_state_dict(state_dict, strict=False) # Allow missing/unexpected keys
                 logger.info(f"Loaded model state. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
            except Exception as e:
                 logger.error(f"Error loading state dict from {state_dict_path}: {e}", exc_info=True)
                 logger.warning("Model weights might be only partially loaded or default initialized.")
        elif is_loading_base_model:
             logger.info(f"Loading base vision encoder weights only. Other layers (RNN, Heads) are randomly initialized.")
        else: # State dict path doesn't exist, and not loading base model
             logger.warning(f"State dict file '{state_dict_path}' not found. Model is using base encoder + random layers.")

        # 4. Ensure model instance has vocabs (copy from config)
        model.base_char_vocab = loaded_config.base_char_vocab
        model.diacritic_vocab = loaded_config.diacritic_vocab
        model.combined_char_vocab = loaded_config.combined_char_vocab

        return model