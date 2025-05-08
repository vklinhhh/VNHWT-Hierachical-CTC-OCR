# model/diacritic_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

class VisualDiacriticAttention(nn.Module):
    """
    Visual attention mechanism that focuses on the areas where diacritics typically appear.
    Creates three attention regions: above, middle, and below the base character.
    Each region has a specialized classifier for diacritics that commonly appear in that position.
    """
    def __init__(self, feature_dim, diacritic_vocab_size):
        super().__init__()
        self.feature_dim = feature_dim
        self.diacritic_vocab_size = diacritic_vocab_size
        
        # Position encoder to generate attention weights for three regions
        self.position_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 3)  # 3 attention maps: above, middle, below
        )
        
        # Define region-specific diacritic classifiers
        # Each classifier focuses on diacritics that typically appear in that region
        self.region_classifiers = nn.ModuleList([
            # Above diacritics (acute, grave, hook, tilde)
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.LayerNorm(feature_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim // 2, diacritic_vocab_size)
            ),
            # Middle diacritics (circumflex, breve, horn)
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.LayerNorm(feature_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim // 2, diacritic_vocab_size)
            ),
            # Below diacritics (dot below)
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.LayerNorm(feature_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim // 2, diacritic_vocab_size)
            )
        ])
        
        # Final fusion layer to combine region-specific predictions
        self.output_fusion = nn.Linear(diacritic_vocab_size * 3, diacritic_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the module."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features):
        """
        Apply visual diacritic attention to input features.
        
        Args:
            features: Tensor of shape [batch_size, seq_length, feature_dim]
                    Typically the output of shared feature layers
        
        Returns:
            diacritic_logits: Tensor of shape [batch_size, seq_length, diacritic_vocab_size]
        """
        batch_size, seq_length, _ = features.shape
        
        # Generate position attention weights
        # [batch_size, seq_length, 3]
        position_logits = self.position_encoder(features)
        position_weights = F.softmax(position_logits, dim=-1)
        
        # Apply each specialized classifier to all positions
        region_logits = []
        for i, classifier in enumerate(self.region_classifiers):
            # [batch_size, seq_length, diacritic_vocab_size]
            region_output = classifier(features)
            
            # Weight by position attention
            # [batch_size, seq_length, 1] * [batch_size, seq_length, diacritic_vocab_size]
            weighted_output = position_weights[:, :, i:i+1] * region_output
            region_logits.append(weighted_output)
        
        # Option 1: Sum the weighted outputs from different regions
        # diacritic_logits = sum(region_logits)
        
        # Option 2: Concatenate and fuse with a linear layer (more expressive)
        # [batch_size, seq_length, diacritic_vocab_size*3]
        concatenated_logits = torch.cat(region_logits, dim=-1)
        # [batch_size, seq_length, diacritic_vocab_size]
        diacritic_logits = self.output_fusion(concatenated_logits)
        
        return diacritic_logits


class CharacterDiacriticCompatibility(nn.Module):
    """
    Explicit Character-Diacritic Compatibility Matrix.
    Models which diacritics are compatible with which base characters,
    e.g., vowels can have tone marks but consonants cannot.
    """
    def __init__(self, base_vocab_size, diacritic_vocab_size, shared_dim=None, 
                 base_char_vocab=None, diacritic_vocab=None):
        """
        Initialize the compatibility module.
        
        Args:
            base_vocab_size: Size of the base character vocabulary
            diacritic_vocab_size: Size of the diacritic vocabulary
            shared_dim: Dimension of shared features (optional)
            base_char_vocab: List of base characters in vocabulary order (optional)
            diacritic_vocab: List of diacritics in vocabulary order (optional)
        """
        super().__init__()
        self.base_vocab_size = base_vocab_size
        self.diacritic_vocab_size = diacritic_vocab_size
        self.base_char_vocab = base_char_vocab
        self.diacritic_vocab = diacritic_vocab
        
        # Learnable compatibility matrix between base chars and diacritics
        # This matrix will learn which combinations are linguistically valid
        self.compatibility_matrix = nn.Parameter(
            torch.zeros(base_vocab_size, diacritic_vocab_size)
        )
        
        # Optional: Create a compatibility predictor that uses character features
        self.use_shared_features = shared_dim is not None
        if self.use_shared_features:
            self.compatibility_predictor = nn.Sequential(
                nn.Linear(shared_dim, shared_dim // 2),
                nn.LayerNorm(shared_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(shared_dim // 2, diacritic_vocab_size)
            )
        
        # Initialize with prior knowledge
        self._initialize_compatibility()
    
    def _initialize_compatibility(self):
        """Initialize the compatibility matrix with linguistic knowledge.
        Uses actual vocabulary indices if available, otherwise uses approximate values.
        """
        # Initialize all possibilities to 0 (neutral)
        with torch.no_grad():
            self.compatibility_matrix.fill_(0.0)
            
            # If we have the actual vocabularies, use them for precise initialization
            if self.base_char_vocab is not None and self.diacritic_vocab is not None:
                # Define vowels and consonants based on Vietnamese alphabet
                vietnamese_vowels = ['a', 'e', 'i', 'o', 'u', 'y', 'A', 'E', 'I', 'O', 'U', 'Y']
                # Special Vietnamese vowels with modifiers but no diacritics
                special_vowels = ['â', 'ê', 'ô', 'ă', 'ơ', 'ư', 'Â', 'Ê', 'Ô', 'Ă', 'Ơ', 'Ư']
                
                # Define tone diacritics
                tone_diacritics = ['acute', 'grave', 'hook', 'tilde', 'dot']
                # Define modifier diacritics
                modifier_diacritics = ['circumflex', 'breve', 'horn']
                # Define combined diacritics
                combined_patterns = [
                    'circumflex_acute', 'circumflex_grave', 'circumflex_hook', 
                    'circumflex_tilde', 'circumflex_dot',
                    'breve_acute', 'breve_grave', 'breve_hook', 
                    'breve_tilde', 'breve_dot',
                    'horn_acute', 'horn_grave', 'horn_hook', 
                    'horn_tilde', 'horn_dot'
                ]
                
                # Log what we found in the vocabularies
                vowel_indices = []
                consonant_indices = []
                special_vowel_indices = []
                tone_indices = []
                modifier_indices = []
                combined_indices = []
                
                # Find vowel and consonant indices in base vocabulary
                for i, char in enumerate(self.base_char_vocab):
                    if char in vietnamese_vowels:
                        vowel_indices.append(i)
                    elif char in special_vowels:
                        special_vowel_indices.append(i)
                    elif char.isalpha() and char not in ['<blank>', '<unk>', '[UNK]']:
                        consonant_indices.append(i)
                
                # Find diacritic type indices
                for i, diac in enumerate(self.diacritic_vocab):
                    if diac in tone_diacritics:
                        tone_indices.append(i)
                    elif diac in modifier_diacritics:
                        modifier_indices.append(i)
                    elif any(pattern in diac for pattern in combined_patterns):
                        combined_indices.append(i)
                
                # Log what we found
                logger.info(f"Found {len(vowel_indices)} vowels, {len(consonant_indices)} consonants, "
                           f"{len(special_vowel_indices)} special vowels in base vocabulary")
                logger.info(f"Found {len(tone_indices)} tone marks, {len(modifier_indices)} modifiers, "
                           f"{len(combined_indices)} combined diacritics in diacritic vocabulary")
                
                # 1. Regular vowels can have tone marks (a -> á, à, ả, ã, ạ)
                for v_idx in vowel_indices:
                    for t_idx in tone_indices:
                        self.compatibility_matrix[v_idx, t_idx] = 2.0
                
                # 2. Special vowels (with modifiers like â, ê, ô, ă, ơ, ư) 
                # can also have tone marks
                for v_idx in special_vowel_indices:
                    for t_idx in tone_indices:
                        self.compatibility_matrix[v_idx, t_idx] = 2.0
                
                # 3. Consonants cannot have tone marks (negative bias)
                for c_idx in consonant_indices:
                    for t_idx in tone_indices:
                        self.compatibility_matrix[c_idx, t_idx] = -2.0
                
                # 4. Handle combined diacritics (these are more specialized)
                # They can only be applied to certain vowels
                
                # Find 'no_diacritic' index (usually index 1 after blank)
                no_diac_idx = None
                for i, diac in enumerate(self.diacritic_vocab):
                    if diac == 'no_diacritic':
                        no_diac_idx = i
                        break
                
                # Set the no-diacritic option to be compatible with everything
                if no_diac_idx is not None:
                    self.compatibility_matrix[:, no_diac_idx] = 1.0
                
                # Find blank character (index 0) and set to prefer blank diacritic
                blank_idx = None
                blank_diac_idx = None
                for i, char in enumerate(self.base_char_vocab):
                    if char == '<blank>':
                        blank_idx = i
                        break
                for i, diac in enumerate(self.diacritic_vocab):
                    if diac == '<blank>':
                        blank_diac_idx = i
                        break
                
                if blank_idx is not None and blank_diac_idx is not None:
                    self.compatibility_matrix[blank_idx, blank_diac_idx] = 3.0
                    # Make blank character incompatible with other diacritics
                    for i in range(self.diacritic_vocab_size):
                        if i != blank_diac_idx:
                            self.compatibility_matrix[blank_idx, i] = -3.0
                
            else:
                # Fallback to approximate indices if vocabularies aren't provided
                logger.warning("Vocabulary lists not provided. Using approximate indices for compatibility matrix.")
                
                # Define vowel indices in the base vocabulary (approximate)
                vowels = [2, 4, 8, 14, 15, 20, 21, 24, 26, 30, 34, 40, 41]  # 'a', 'e', 'i', 'o', 'u', 'y', etc.
                
                # Define tone mark indices in the diacritic vocabulary (approximate)
                tone_marks = [3, 4, 5, 6, 7]  # acute, grave, hook, tilde, dot
                
                # Set vowel-tone combinations to higher initial values (e.g., 2.0)
                for v in vowels:
                    for t in tone_marks:
                        if v < self.base_vocab_size and t < self.diacritic_vocab_size:
                            self.compatibility_matrix[v, t] = 2.0
                
                # Set consonant-tone combinations to lower initial values (e.g., -2.0)
                # This discourages the model from predicting tone marks on consonants
                consonants = [i for i in range(self.base_vocab_size) if i not in vowels and i >= 2]
                for c in consonants:
                    for t in tone_marks:
                        if c < self.base_vocab_size and t < self.diacritic_vocab_size:
                            self.compatibility_matrix[c, t] = -2.0
                
                # Set the no-diacritic option (usually index 1 after blank) to be compatible with everything
                if 1 < self.diacritic_vocab_size:
                    self.compatibility_matrix[:, 1] = 1.0
                
                # Set blank character (index 0) to prefer blank diacritic
                if 0 < self.base_vocab_size and 0 < self.diacritic_vocab_size:
                    self.compatibility_matrix[0, 0] = 3.0
                    self.compatibility_matrix[0, 1:] = -3.0
        
        logger.info("Initialized character-diacritic compatibility matrix with linguistic priors")
    
    def forward(self, base_logits, shared_features=None):
        """
        Apply compatibility constraints between base characters and diacritics.
        
        Args:
            base_logits: Tensor of shape [batch_size, seq_length, base_vocab_size]
                        Logits from the base character classifier
            shared_features: Optional tensor of shape [batch_size, seq_length, shared_dim]
                            Shared features for additional compatibility prediction
        
        Returns:
            compatibility_bias: Tensor of shape [batch_size, seq_length, diacritic_vocab_size]
                               Bias to add to diacritic logits based on compatibility
        """
        # Convert base logits to probabilities
        base_probs = F.softmax(base_logits, dim=-1)
        
        # Calculate compatibility bias using matrix multiplication
        # [batch_size, seq_length, base_vocab_size] × [base_vocab_size, diacritic_vocab_size]
        # -> [batch_size, seq_length, diacritic_vocab_size]
        compatibility_bias = torch.matmul(base_probs, self.compatibility_matrix)
        
        # If using shared features, add additional compatibility prediction
        if self.use_shared_features and shared_features is not None:
            additional_bias = self.compatibility_predictor(shared_features)
            compatibility_bias = compatibility_bias + additional_bias
        
        return compatibility_bias


class FewShotDiacriticAdapter(nn.Module):
    """
    Few-Shot Diacritic Learning adapter for rare diacritic combinations.
    Uses a prototype-based approach to generalize from few examples.
    """
    def __init__(self, feature_dim, diacritic_vocab_size, num_prototypes=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.diacritic_vocab_size = diacritic_vocab_size
        self.num_prototypes = num_prototypes
        
        # Learnable prototypes for each diacritic class
        # These will be updated during training to represent typical feature patterns
        self.prototypes = nn.Parameter(
            torch.randn(diacritic_vocab_size, num_prototypes, feature_dim)
        )
        
        # Scaling factor for prototype matching (temperature parameter)
        self.temperature = nn.Parameter(torch.tensor(10.0))
        
        # Optional: attention mechanism to focus on relevant feature dimensions
        self.feature_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Linear(diacritic_vocab_size, diacritic_vocab_size)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the module."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize prototypes to have unit norm
        with torch.no_grad():
            normalized_prototypes = F.normalize(self.prototypes, p=2, dim=-1)
            self.prototypes.copy_(normalized_prototypes)
    
    def forward(self, features):
        """
        Apply few-shot diacritic learning to input features.
        
        Args:
            features: Tensor of shape [batch_size, seq_length, feature_dim]
                    Typically the output of shared feature layers
        
        Returns:
            diacritic_logits: Tensor of shape [batch_size, seq_length, diacritic_vocab_size]
        """
        batch_size, seq_length, _ = features.shape
        
        # Apply feature attention
        attention_weights = self.feature_attention(features)
        weighted_features = features * attention_weights
        
        # Normalize features for cosine similarity
        normalized_features = F.normalize(weighted_features, p=2, dim=-1)
        
        # Calculate similarity to each prototype
        # For efficiency, reshape and compute all similarities at once
        # Reshape prototypes: [diacritic_vocab_size, num_prototypes, feature_dim]
        # -> [diacritic_vocab_size * num_prototypes, feature_dim]
        flat_prototypes = self.prototypes.view(-1, self.feature_dim)
        
        # Reshape features: [batch_size, seq_length, feature_dim]
        # -> [batch_size * seq_length, feature_dim]
        flat_features = normalized_features.view(-1, self.feature_dim)
        
        # Compute similarities
        # [batch_size * seq_length, diacritic_vocab_size * num_prototypes]
        similarities = torch.matmul(flat_features, flat_prototypes.transpose(0, 1))
        
        # Reshape similarities to [batch_size, seq_length, diacritic_vocab_size, num_prototypes]
        similarities = similarities.view(batch_size, seq_length, self.diacritic_vocab_size, self.num_prototypes)
        
        # Take max similarity for each class (max-pooling over prototypes)
        # [batch_size, seq_length, diacritic_vocab_size]
        max_similarities, _ = torch.max(similarities, dim=-1)
        
        # Apply temperature scaling
        scaled_similarities = max_similarities * self.temperature
        
        # Optional: Apply output projection for better expressivity
        diacritic_logits = self.output_projection(scaled_similarities)
        
        return diacritic_logits