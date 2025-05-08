#!/usr/bin/env python
# scripts/demo_enhanced_diacritics.py
"""
Demo script for visualizing the enhanced diacritic classifier approaches.
Loads a hierarchical model with different diacritic enhancement options and
shows their impact on Vietnamese character recognition.
"""

import os
import sys
import argparse
import torch
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.colors import LinearSegmentedColormap

from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel, HierarchicalCtcOcrConfig
from utils.ctc_utils import CTCDecoder

# Set up basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('demo_enhanced_diacritics')


def create_heatmap_overlay(image, attention_weights, overlay_alpha=0.5, cmap='hot'):
    """Create an overlay of attention weights on the original image."""
    # Resize attention weights to match image size
    from PIL import Image
    import numpy as np
    
    # Convert attention weights to numpy array if it's a tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # Normalize attention weights to [0, 1]
    attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-9)
    
    # Convert image to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
    
    # Resize attention weights to match image dimensions
    h, w = image_array.shape[:2]
    attention_resized = np.zeros((h, w))
    
    # Simple resize (for demonstration)
    # In a real implementation, you would use a proper interpolation method
    ah, aw = attention_weights.shape
    for i in range(h):
        for j in range(w):
            # Map image coordinates to attention coordinates
            ai = min(int(i * ah / h), ah - 1)
            aj = min(int(j * aw / w), aw - 1)
            attention_resized[i, j] = attention_weights[ai, aj]
    
    # Create colormap
    cmap_fn = plt.get_cmap(cmap)
    attention_color = cmap_fn(attention_resized)
    attention_color = attention_color[:, :, :3]  # Remove alpha channel
    
    # Create overlay
    overlay = image_array.copy().astype(float)
    for c in range(3):  # RGB channels
        overlay[:, :, c] = image_array[:, :, c] * (1 - overlay_alpha) + attention_color[:, :, c] * 255 * overlay_alpha
    
    return overlay.astype(np.uint8)

def visualize_model_predictions(image_path, model, decoder, device, output_dir=None):
    """Visualize model predictions and attention weights."""
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    
    # Get image dimensions for display
    width, height = image.size
    aspect_ratio = width / height
    
    # Process image with model processor
    pixel_values = model.processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        
        # Get outputs
        final_logits = outputs['logits']
        base_logits = outputs['base_logits']
        diacritic_logits = outputs['diacritic_logits']
        
        # Decode predictions
        final_preds = decoder(final_logits)
        
    # Get predicted text
    predicted_text = final_preds[0]
    logger.info(f"Predicted text: {predicted_text}")
    
    # Create figure for visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot original image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot base character probabilities 
    # (Take the average across sequence positions for visualization)
    base_probs = torch.softmax(base_logits[0], dim=-1).cpu().numpy()
    diacritic_probs = torch.softmax(diacritic_logits[0], dim=-1).cpu().numpy()
    
    # Create attention maps
    # For Visual Diacritic Attention, we would want to see where it's focusing
    # Since we don't have direct access to the attention weights in the forward pass,
    # we'll simulate them based on the feature representations
    
    # For base characters - use the max probability across base characters
    base_attention = base_probs.max(axis=1)
    # Reshape to approximate 2D structure of the original image
    seq_len = base_attention.shape[0]
    width_factor = int(np.sqrt(seq_len * aspect_ratio))
    height_factor = int(seq_len / width_factor)
    
    # Adjust for edge cases
    while width_factor * height_factor < seq_len:
        height_factor += 1
        
    # Reshape attention to 2D grid
    base_attention_2d = np.zeros((height_factor, width_factor))
    idx = 0
    for i in range(height_factor):
        for j in range(width_factor):
            if idx < seq_len:
                base_attention_2d[i, j] = base_attention[idx]
                idx += 1
    
    # For diacritics - focus on regions with high diacritic probabilities
    # Skip the blank/no_diacritic classes (indices 0 and 1)
    if diacritic_probs.shape[1] > 2:
        diacritic_attention = diacritic_probs[:, 2:].max(axis=1)  # Skip blank/no_diacritic
        
        # Reshape to 2D
        diacritic_attention_2d = np.zeros((height_factor, width_factor))
        idx = 0
        for i in range(height_factor):
            for j in range(width_factor):
                if idx < seq_len:
                    diacritic_attention_2d[i, j] = diacritic_attention[idx]
                    idx += 1
    else:
        diacritic_attention_2d = np.zeros((height_factor, width_factor))
    
    # Create attention visualizations
    plt.subplot(2, 2, 2)
    plt.imshow(base_attention_2d, cmap='viridis')
    plt.title("Base Character Attention")
    plt.colorbar(shrink=0.8)
    
    plt.subplot(2, 2, 3)
    plt.imshow(diacritic_attention_2d, cmap='plasma')
    plt.title("Diacritic Attention")
    plt.colorbar(shrink=0.8)
    
    # Create combined visualization (overlay on original image)
    plt.subplot(2, 2, 4)
    
    # Normalize and resize attention maps to match image size
    combined_attention = 0.7 * base_attention_2d + 0.3 * diacritic_attention_2d
    
    # Create an RGB image from combined attention
    cmap = plt.get_cmap('viridis')
    combined_attention_rgb = cmap(combined_attention)[:, :, :3]  # Remove alpha channel
    
    # Convert to PIL image and resize
    combined_att_image = Image.fromarray((combined_attention_rgb * 255).astype(np.uint8))
    combined_att_image = combined_att_image.resize(
        (width, height), 
        Image.BICUBIC
    )
    combined_att_array = np.array(combined_att_image)
    
    # Create a new figure for the overlay
    overlay_alpha = 0.5
    # Convert the original image to numpy array if it's not already
    image_array = np.array(image)
    overlay = image_array * (1 - overlay_alpha) + combined_att_array * overlay_alpha
    plt.imshow(overlay.astype(np.uint8))
    plt.title(f"Prediction: {predicted_text}")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save or show the figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, "enhanced_diacritics_demo.png"))
        logger.info(f"Saved visualization to {output_dir}/enhanced_diacritics_demo.png")
    else:
        plt.show()
    
    # Create another visualization specifically comparing the three enhancement approaches
    plt.figure(figsize=(15, 10))
    
    # Plot original image with prediction
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title(f"Original: {predicted_text}")
    plt.axis('off')
    
    # Add annotations to show which enhancements are enabled
    enhancement_text = ""
    if hasattr(model.config, 'use_visual_diacritic_attention') and model.config.use_visual_diacritic_attention:
        enhancement_text += "✓ Visual Attention\n"
    else:
        enhancement_text += "✗ Visual Attention\n"
        
    if hasattr(model.config, 'use_character_diacritic_compatibility') and model.config.use_character_diacritic_compatibility:
        enhancement_text += "✓ Char-Diacritic Compatibility\n"
    else:
        enhancement_text += "✗ Char-Diacritic Compatibility\n"
        
    if hasattr(model.config, 'use_few_shot_diacritic_adapter') and model.config.use_few_shot_diacritic_adapter:
        enhancement_text += f"✓ Few-Shot Adapter ({model.config.num_few_shot_prototypes} prototypes)"
    else:
        enhancement_text += "✗ Few-Shot Adapter"
    
    plt.figtext(0.15, 0.4, enhancement_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Visualize diacritic-specific features
    
    # 1. Display base character probabilities for a few common characters
    if hasattr(model, 'base_char_vocab') and model.base_char_vocab:
        base_vocab = model.base_char_vocab[:20]  # First 20 chars
    else:
        base_vocab = [f"Char_{i}" for i in range(20)]
        
    plt.subplot(2, 2, 2)
    
    # Get average probabilities across sequence positions
    avg_base_probs = base_probs.mean(axis=0)[:20]  # First 20 classes
    plt.bar(range(len(avg_base_probs)), avg_base_probs)
    plt.xticks(range(len(avg_base_probs)), base_vocab, rotation=45)
    plt.title("Base Character Probabilities")
    plt.tight_layout()
    
    # 2. Display diacritic probabilities
    plt.subplot(2, 2, 3)
    
    # Get average probabilities across sequence
    if hasattr(model, 'diacritic_vocab') and model.diacritic_vocab:
        diacritic_vocab = model.diacritic_vocab[:15]  # First 15 diacritics
    else:
        diacritic_vocab = [f"Diac_{i}" for i in range(15)]
        
    avg_diac_probs = diacritic_probs.mean(axis=0)[:15]  # First 15 classes
    plt.bar(range(len(avg_diac_probs)), avg_diac_probs)
    plt.xticks(range(len(avg_diac_probs)), diacritic_vocab, rotation=45)
    plt.title("Diacritic Probabilities")
    plt.tight_layout()
    
    # 3. Display sequence-level visualization
    plt.subplot(2, 2, 4)
    
    # Create sequence visualization (characters vs positions)
    seq_length = min(20, base_probs.shape[0])  # First 20 positions
    
    # For each position, find the most likely character and diacritic
    char_indices = base_probs[:seq_length].argmax(axis=1)
    diac_indices = diacritic_probs[:seq_length].argmax(axis=1)
    
    # Get characters and diacritics
    if hasattr(model, 'base_char_vocab') and model.base_char_vocab:
        chars = [model.base_char_vocab[i] if i < len(model.base_char_vocab) else '?' for i in char_indices]
    else:
        chars = [f"C{i}" for i in char_indices]
        
    if hasattr(model, 'diacritic_vocab') and model.diacritic_vocab:
        diacs = [model.diacritic_vocab[i] if i < len(model.diacritic_vocab) else '?' for i in diac_indices]
    else:
        diacs = [f"D{i}" for i in diac_indices]
    
    # Create a formatted string representation
    characters = []
    for pos in range(seq_length):
        char = chars[pos]
        diac = diacs[pos]
        
        # Skip blanks and duplicates (CTC behavior)
        if char == '<blank>' or (pos > 0 and char == chars[pos-1] and char != ' '):
            continue
            
        # Format character with diacritic information
        if diac == 'no_diacritic' or diac == '<blank>':
            characters.append(char)
        else:
            characters.append(f"{char}+{diac}")
    
    # Display the sequence
    plt.text(0.1, 0.5, ' '.join(characters), fontsize=12)
    plt.axis('off')
    plt.title("Character Sequence Breakdown")
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        plt.savefig(os.path.join(output_dir, "enhanced_diacritics_analysis.png"))
        logger.info(f"Saved analysis to {output_dir}/enhanced_diacritics_analysis.png")
    else:
        plt.show()
    
    # Close plots
    plt.close('all')
    
    return predicted_text


def main():
    parser = argparse.ArgumentParser(description="Demo for Enhanced Diacritic Classifiers")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained hierarchical model with diacritic enhancements")
    parser.add_argument("--combined_char_vocab_path", type=str, required=True,
                       help="Path to combined character vocabulary JSON file")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to Vietnamese handwritten text image to analyze")
    parser.add_argument("--output_dir", type=str, default="diacritic_demo_output",
                       help="Directory to save visualization outputs")
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load vocabulary
    try:
        logger.info(f"Loading vocabulary from {args.combined_char_vocab_path}")
        with open(args.combined_char_vocab_path, 'r', encoding='utf-8') as f:
            combined_vocab = json.load(f)
        
        # Create character-to-index mapping for decoder
        combined_idx_to_char = {i: c for i, c in enumerate(combined_vocab)}
        blank_idx = 0  # Assuming blank is at index 0
    except Exception as e:
        logger.error(f"Failed to load vocabulary: {e}")
        return 1
    
    # Load model
    try:
        logger.info(f"Loading model from {args.model_path}")
        
        # First load the vocabulary files directly
        base_vocab_path = os.path.join(args.model_path, "base_char_vocab.json")
        diacritic_vocab_path = os.path.join(args.model_path, "diacritic_vocab.json")
        
        # Check if separate vocabulary files exist
        base_char_vocab = None
        diacritic_vocab = None
        
        if os.path.exists(base_vocab_path):
            try:
                with open(base_vocab_path, 'r', encoding='utf-8') as f:
                    base_char_vocab = json.load(f)
                logger.info(f"Loaded base character vocabulary with {len(base_char_vocab)} entries")
            except Exception as e:
                logger.warning(f"Could not load base character vocabulary: {e}")
        
        if os.path.exists(diacritic_vocab_path):
            try:
                with open(diacritic_vocab_path, 'r', encoding='utf-8') as f:
                    diacritic_vocab = json.load(f)
                logger.info(f"Loaded diacritic vocabulary with {len(diacritic_vocab)} entries")
            except Exception as e:
                logger.warning(f"Could not load diacritic vocabulary: {e}")
        
        # Load the model with vocabularies
        model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(
            args.model_path,
            combined_char_vocab=combined_vocab,
            base_char_vocab=base_char_vocab,
            diacritic_vocab=diacritic_vocab
        )
        model.to(device)
        
        # Save vocabularies to model for later use if they're missing
        if hasattr(model, 'base_char_vocab') and model.base_char_vocab and base_char_vocab is None:
            # Save the vocabulary for next time
            with open(base_vocab_path, 'w', encoding='utf-8') as f:
                json.dump(model.base_char_vocab, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved base character vocabulary to {base_vocab_path}")
        
        if hasattr(model, 'diacritic_vocab') and model.diacritic_vocab and diacritic_vocab is None:
            # Save the vocabulary for next time
            with open(diacritic_vocab_path, 'w', encoding='utf-8') as f:
                json.dump(model.diacritic_vocab, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved diacritic vocabulary to {diacritic_vocab_path}")
        
        # Log which diacritic enhancements are enabled
        logger.info("Model configuration:")
        if hasattr(model.config, 'use_visual_diacritic_attention'):
            logger.info(f"Visual Diacritic Attention: {'Enabled' if model.config.use_visual_diacritic_attention else 'Disabled'}")
        else:
            logger.info("Visual Diacritic Attention: Not configured (disabled)")
            
        if hasattr(model.config, 'use_character_diacritic_compatibility'):
            logger.info(f"Character-Diacritic Compatibility: {'Enabled' if model.config.use_character_diacritic_compatibility else 'Disabled'}")
        else:
            logger.info("Character-Diacritic Compatibility: Not configured (disabled)")
            
        if hasattr(model.config, 'use_few_shot_diacritic_adapter'):
            logger.info(f"Few-Shot Diacritic Adapter: {'Enabled' if model.config.use_few_shot_diacritic_adapter else 'Disabled'}")
            if model.config.use_few_shot_diacritic_adapter:
                logger.info(f"Number of prototypes: {model.config.num_few_shot_prototypes}")
        else:
            logger.info("Few-Shot Diacritic Adapter: Not configured (disabled)")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Initialize decoder
    decoder = CTCDecoder(idx_to_char_map=combined_idx_to_char, blank_idx=blank_idx)
    
    # Run visualization
    logger.info(f"Processing image: {args.image_path}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        predicted_text = visualize_model_predictions(
            args.image_path, model, decoder, device, args.output_dir
        )
        
        logger.info("Demo completed successfully")
        logger.info(f"Predicted text: {predicted_text}")
        logger.info(f"Visualizations saved to: {args.output_dir}")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)