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
    model_type = "hierarchical_ctc_ocr_multiscale"

    def __init__(
        self,
        vision_encoder_name='microsoft/trocr-base-handwritten',
        # Vocabs needed for head sizes and potentially conditioning
        base_char_vocab=None, # For size of base head
        diacritic_vocab=None, # For size of potential diacritic head / conditioning
        combined_char_vocab=None, # Combined chars 'a','á','à' for FINAL CTC output
        vision_encoder_layer_indices=[-1, -4], # Indices of layers to fuse (e.g., last and 4th-to-last)
                                               # Adjust based on encoder architecture (e.g., ViT has 12/24 layers)
        feature_fusion_method="concat_proj", # Options: 'concat_proj', 'add', 'bilinear'
        # Intermediate layers
        intermediate_rnn_layers=2,
        rnn_hidden_size=512,
        rnn_dropout=0.1,
        rnn_bidirectional=True,
        # Shared / Hierarchical Layers
        shared_hidden_size=512, # Dimension after RNN, before branching
        num_shared_layers=2, 
        conditioning_method="concat_proj", # 'concat_proj', 'gate', 'none'
        # Classifier dropout
        classifier_dropout=0.2,
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
        if not isinstance(vision_encoder_layer_indices, list) or len(vision_encoder_layer_indices) < 2:
             logger.warning("vision_encoder_layer_indices must be a list of at least two indices. Defaulting to [-1, -4].")
             self.vision_encoder_layer_indices = [-1, -4]
        else:
             self.vision_encoder_layer_indices = sorted(list(set(vision_encoder_layer_indices))) # Sort and unique
        self.feature_fusion_method = feature_fusion_method
        self.intermediate_rnn_layers = intermediate_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_dropout = rnn_dropout
        self.rnn_bidirectional = rnn_bidirectional
        self.shared_hidden_size = shared_hidden_size
        self.num_shared_layers = num_shared_layers 
        self.conditioning_method = conditioning_method
        self.classifier_dropout = classifier_dropout
        self.blank_idx = blank_idx
        self.vision_encoder_config = vision_encoder_config # Store this too

        if not self.combined_char_vocab:
            # Allow init without combined if loading base, but log warning
            logger.warning("Combined character vocabulary is empty during config init.")
        # No error needed here, check happens in model __init__


# --- Model Class ---
class HierarchicalCtcMultiScaleOcrModel(PreTrainedModel):
    config_class = HierarchicalCtcOcrConfig

    def __init__(self, config: HierarchicalCtcOcrConfig):
        super().__init__(config)
        self.config = config # Store config on the model instance

        # *** Ensure vocabs needed for layer sizes are present ***
        if not config.combined_char_vocab:
            raise ValueError("Combined character vocabulary must be provided in the config during model initialization.")
        if config.conditioning_method != 'none' and (not config.base_char_vocab or not config.diacritic_vocab):
             raise ValueError("Base/Diacritic vocabs must be provided in config for conditioning methods 'concat_proj' or 'gate'.")

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
            
        # --- Feature Fusion Layer ---
        encoder_output_size = self.config.vision_encoder_config.hidden_size
        num_fusion_layers = len(config.vision_encoder_layer_indices)
        fusion_input_size = encoder_output_size * num_fusion_layers # Default for concat

        if config.feature_fusion_method == "concat_proj":
            # Project concatenated features back to original encoder size or RNN input size
            self.fusion_projection = nn.Linear(fusion_input_size, encoder_output_size)
            rnn_input_size = encoder_output_size # RNN takes projected features
            logger.info(f"Using 'concat_proj' feature fusion (In: {fusion_input_size}, Out: {rnn_input_size})")
        elif config.feature_fusion_method == "add":
            # Features must have the same dimension for addition
            rnn_input_size = encoder_output_size
            self.fusion_projection = None # No projection needed for simple add
            logger.info("Using 'add' feature fusion")
        elif config.feature_fusion_method == "bilinear":
            # Bilinear pooling (more complex, example placeholder)
            # Output size might differ, e.g., might be encoder_output_size
            self.fusion_bilinear = nn.Bilinear(encoder_output_size, encoder_output_size, encoder_output_size) # Example
            rnn_input_size = encoder_output_size
            self.fusion_projection = None
            logger.info("Using 'bilinear' feature fusion (Example implementation)")
        else: # Default to no fusion or just last layer
            logger.warning(f"Unknown feature_fusion_method: {config.feature_fusion_method}. Using only last encoder layer.")
            self.config.feature_fusion_method = "none"
            self.fusion_projection = None
            rnn_input_size = encoder_output_size # RNN takes last layer output

        # --- Intermediate Layers ---
        logger.info(f"Adding {config.intermediate_rnn_layers} RNN layers (Input Size: {rnn_input_size})...")
        self.rnn = nn.GRU(
            input_size=encoder_output_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.intermediate_rnn_layers,
            batch_first=True,
            dropout=config.rnn_dropout if config.intermediate_rnn_layers > 1 else 0,
            bidirectional=config.rnn_bidirectional
        )
        rnn_output_size = config.rnn_hidden_size * 2 if config.rnn_bidirectional else config.rnn_hidden_size

        # --- Deeper Shared Feature Layer ---
        shared_layers = []
        current_shared_size = rnn_output_size
        for i in range(config.num_shared_layers):
            shared_layers.extend(
                [
                    nn.Linear(current_shared_size, config.shared_hidden_size),
                    nn.LayerNorm(config.shared_hidden_size),
                    nn.GELU(),
                    nn.Dropout(config.classifier_dropout),
                ]
            )
            current_shared_size = (
                config.shared_hidden_size
            )  # Subsequent layers use shared_hidden_size
        self.shared_layer = nn.Sequential(*shared_layers)
        logger.info(
            f'Added {config.num_shared_layers} shared feature layer(s) (Final Output: {config.shared_hidden_size})'
        )

        # --- Hierarchical Branching ---
        self.base_classifier = nn.Linear(config.shared_hidden_size, config.base_char_vocab_size)
        logger.info(f"Added Base Classifier Head (Output: {config.base_char_vocab_size})")
        
        # --- Conditioning Logic & Diacritic Head ---
        conditioning_input_size = config.shared_hidden_size
        self.diacritic_gate = None
        self.diacritic_condition_proj = None  # Initialize projection layer

        if config.conditioning_method == 'concat_proj':  # Renamed 'concat' to 'concat_proj'
            concat_size = (
                config.shared_hidden_size + config.base_char_vocab_size
            )  # Shared feats + Base Logits
            # Add projection layer
            self.diacritic_condition_proj = nn.Sequential(
                nn.Linear(concat_size, config.shared_hidden_size),  # Project back down
                nn.LayerNorm(config.shared_hidden_size),
                nn.GELU(),
                nn.Dropout(config.classifier_dropout),
            )
            diacritic_head_input_size = config.shared_hidden_size  # Head takes projected features
            logger.info("Using 'concat_proj' conditioning for diacritic head.")
        elif config.conditioning_method == 'gate':
            self.diacritic_gate = nn.Sequential(
                nn.Linear(config.shared_hidden_size + config.base_char_vocab_size, config.shared_hidden_size),
                nn.Sigmoid()
            )
            diacritic_head_input_size = config.shared_hidden_size  # Head takes gated features
            logger.info("Using 'gate' conditioning for diacritic head.")
        else:  # 'none' or unknown
            if config.conditioning_method != 'none':
                logger.warning(
                    f"Unknown conditioning method '{config.conditioning_method}'. Defaulting to 'none'."
                )
                self.config.conditioning_method = 'none'
            diacritic_head_input_size = (
                config.shared_hidden_size
            )  # Head takes shared features directly
            logger.info("Using 'none' conditioning for diacritic head.")

        self.diacritic_classifier = nn.Linear(
            diacritic_head_input_size, config.diacritic_vocab_size
        )
        logger.info(
            f'Added Diacritic Classifier Head (Input: {diacritic_head_input_size}, Output: {config.diacritic_vocab_size})'
        )

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
        # *** Init Fusion Projection Layer ***
        if hasattr(self, 'fusion_projection') and self.fusion_projection is not None:
            nn.init.xavier_uniform_(self.fusion_projection.weight)
            if self.fusion_projection.bias is not None: nn.init.zeros_(self.fusion_projection.bias)
        # Projection Layer
        if self.diacritic_condition_proj:
            for layer in self.diacritic_condition_proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        pixel_values,
        labels=None,
        label_lengths=None
        ):
        # 1. Vision Encoder (Get Multiple Hidden States)
        encoder_outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True, # <<< Request hidden states
            return_dict=True
        )
        all_hidden_states = encoder_outputs.hidden_states # Tuple of hidden states from embedding + each layer

        # 2. Select and Fuse Features
        # Ensure indices are valid for the number of layers available
        num_encoder_layers = len(all_hidden_states) - 1 # Exclude embedding output
        valid_indices = []
        for idx in self.config.vision_encoder_layer_indices:
            actual_idx = idx if idx >= 0 else num_encoder_layers + 1 + idx # Handle negative indices
            if 0 < actual_idx <= num_encoder_layers + 1 : # 0 is embedding, 1 to N are layers
                 valid_indices.append(actual_idx)
            else:
                 logger.warning(f"Invalid encoder layer index {idx} ignored.")

        if len(valid_indices) < 1: # Need at least one layer
             logger.warning("No valid encoder layers selected for fusion, using only last layer.")
             features_to_fuse = [all_hidden_states[-1]] # Fallback to last layer
        else:
             features_to_fuse = [all_hidden_states[i] for i in valid_indices]


        # Apply fusion method
        if self.config.feature_fusion_method == "concat_proj" and len(features_to_fuse) > 1:
            concatenated_features = torch.cat(features_to_fuse, dim=-1)
            fused_features = self.fusion_projection(concatenated_features)
        elif self.config.feature_fusion_method == "add" and len(features_to_fuse) > 1:
            # Simple averaging or summing (ensure dimensions match)
            fused_features = torch.stack(features_to_fuse, dim=0).mean(dim=0)
        elif self.config.feature_fusion_method == "bilinear" and len(features_to_fuse) == 2:
             # Example bilinear (requires exactly 2 features)
             fused_features = self.fusion_bilinear(features_to_fuse[0], features_to_fuse[1])
        else: # 'none' or fallback
            fused_features = features_to_fuse[-1] # Use the last selected layer (usually the final encoder output)


        # 3. RNN Layers (Input is now fused_features)
        rnn_outputs, _ = self.rnn(fused_features) # [B, T_rnn, D_rnn*dirs]

        # 4. Shared Feature Layer
        shared_features = self.shared_layer(rnn_outputs)  # [B, T, D_shared]

        # 5. Hierarchical Heads & Final Classifier (Same logic as before)
        base_logits = self.base_classifier(shared_features) # [B, T, N_base]
        
        if self.config.conditioning_method == 'concat_proj':
            # Concatenate then project
            concat_features = torch.cat((shared_features, base_logits), dim=-1)
            diacritic_input_features = self.diacritic_condition_proj(
                concat_features
            )  # Use projected
        elif self.config.conditioning_method == 'gate' and self.diacritic_gate is not None:
            # Gate the shared features
            gate_input = torch.cat((shared_features, base_logits), dim=-1)
            diacritic_gate_values = self.diacritic_gate(gate_input)
            diacritic_input_features = shared_features * diacritic_gate_values  # Apply gate
        else:  # 'none'
            diacritic_input_features = shared_features  # Use shared directly

        diacritic_logits = self.diacritic_classifier(diacritic_input_features)  # [B, T, N_diac]

        # 6. Final Combined Classifier
        final_input_features = shared_features
        final_logits = self.final_classifier(final_input_features)

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
        if hasattr(self.vision_encoder, 'config'): self.config.vision_encoder_config = self.vision_encoder.config
        
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

        logger.info(f"Loading {cls.__name__} from: {pretrained_model_name_or_path}")
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        loaded_config = None

        if config is not None and isinstance(config, cls.config_class): loaded_config = config; logger.info("Using provided config object.")
        elif os.path.exists(config_path): loaded_config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs); logger.info(f"Loading config from file: {config_path}")
        else: # Initialize new config from kwargs
            logger.warning(f"Config file not found at {config_path}. Initializing new config.")
            if not all(k in kwargs for k in ['base_char_vocab', 'diacritic_vocab', 'combined_char_vocab']): raise ValueError("All vocabs required in kwargs w/o config.json.")
            if 'vision_encoder_name' not in kwargs: kwargs['vision_encoder_name'] = cls.config_class().vision_encoder_name # Get default vision name
            loaded_config = cls.config_class(**kwargs)

        # Override vocabs from kwargs if provided AFTER loading/creating config
        if 'base_char_vocab' in kwargs and kwargs['base_char_vocab']: loaded_config.base_char_vocab = kwargs['base_char_vocab']; loaded_config.base_char_vocab_size = len(kwargs['base_char_vocab'])
        if 'diacritic_vocab' in kwargs and kwargs['diacritic_vocab']: loaded_config.diacritic_vocab = kwargs['diacritic_vocab']; loaded_config.diacritic_vocab_size = len(kwargs['diacritic_vocab'])
        if 'combined_char_vocab' in kwargs and kwargs['combined_char_vocab']: loaded_config.combined_char_vocab = kwargs['combined_char_vocab']; loaded_config.combined_char_vocab_size = len(kwargs['combined_char_vocab'])

        model = cls(loaded_config) # Instantiate with final config

        # Load state dict
        state_dict_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        is_loading_base_model = not os.path.exists(config_path)
        if os.path.exists(state_dict_path) and not is_loading_base_model:
            try: state_dict = torch.load(state_dict_path, map_location="cpu"); load_result = model.load_state_dict(state_dict, strict=False); logger.info(f"Loaded state. Miss:{load_result.missing_keys}, Unexp:{load_result.unexpected_keys}")
            except Exception as e: logger.error(f"Error loading state dict: {e}")
        elif is_loading_base_model: logger.info(f"Loading base weights only.")
        else: logger.warning(f"State dict not found. Using base + random.")

        model.base_char_vocab = loaded_config.base_char_vocab; model.diacritic_vocab = loaded_config.diacritic_vocab; model.combined_char_vocab = loaded_config.combined_char_vocab
        return model