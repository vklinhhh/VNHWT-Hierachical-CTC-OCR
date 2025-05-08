#!/usr/bin/env python
# scripts/extract_vocabularies.py
"""
Utility script to extract and save vocabularies from a trained hierarchical model.
This can fix issues with missing or incomplete vocabulary files.
"""

import os
import sys
import argparse
import json
import logging
import torch

from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('extract_vocabularies')

def extract_vocabularies(model_path, output_path=None):
    """
    Extract vocabularies from a trained model and save them as JSON files.
    
    Args:
        model_path: Path to the model directory
        output_path: Path to save vocabularies (defaults to model_path if None)
    
    Returns:
        0 on success, 1 on failure
    """
    if output_path is None:
        output_path = model_path
    
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # First check if config.json exists and try to extract vocab lists
        config_path = os.path.join(model_path, "config.json")
        combined_vocab = None
        base_vocab = None
        diacritic_vocab = None
        
        if os.path.exists(config_path):
            logger.info(f"Loading config from {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Extract vocabularies from config
            combined_vocab = config.get('combined_char_vocab', None)
            base_vocab = config.get('base_char_vocab', None)
            diacritic_vocab = config.get('diacritic_vocab', None)
            
            if combined_vocab:
                logger.info(f"Found combined vocabulary in config ({len(combined_vocab)} entries)")
            if base_vocab:
                logger.info(f"Found base vocabulary in config ({len(base_vocab)} entries)")
            if diacritic_vocab:
                logger.info(f"Found diacritic vocabulary in config ({len(diacritic_vocab)} entries)")
        
        # If we still need vocabularies, try loading the model
        if not all([combined_vocab, base_vocab, diacritic_vocab]):
            logger.info("Loading model to extract vocabularies")
            model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(model_path)
            
            # Extract vocabularies from model attributes
            if not combined_vocab and hasattr(model, 'combined_char_vocab'):
                combined_vocab = model.combined_char_vocab
                logger.info(f"Extracted combined vocabulary from model ({len(combined_vocab)} entries)")
            
            if not base_vocab and hasattr(model, 'base_char_vocab'):
                base_vocab = model.base_char_vocab
                logger.info(f"Extracted base vocabulary from model ({len(base_vocab)} entries)")
            
            if not diacritic_vocab and hasattr(model, 'diacritic_vocab'):
                diacritic_vocab = model.diacritic_vocab
                logger.info(f"Extracted diacritic vocabulary from model ({len(diacritic_vocab)} entries)")
        
        # Save vocabularies
        success = False
        
        if combined_vocab:
            combined_path = os.path.join(output_path, "combined_char_vocab.json")
            with open(combined_path, 'w', encoding='utf-8') as f:
                json.dump(combined_vocab, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved combined vocabulary to {combined_path}")
            success = True
        
        if base_vocab:
            base_path = os.path.join(output_path, "base_char_vocab.json")
            with open(base_path, 'w', encoding='utf-8') as f:
                json.dump(base_vocab, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved base vocabulary to {base_path}")
            success = True
        
        if diacritic_vocab:
            diacritic_path = os.path.join(output_path, "diacritic_vocab.json")
            with open(diacritic_path, 'w', encoding='utf-8') as f:
                json.dump(diacritic_vocab, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved diacritic vocabulary to {diacritic_path}")
            success = True
        
        if not success:
            logger.error("Could not extract any vocabularies")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to extract vocabularies: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Extract vocabularies from a trained hierarchical model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model directory")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save vocabularies (defaults to model_path if not specified)")
    
    args = parser.parse_args()
    
    return extract_vocabularies(args.model_path, args.output_path)

if __name__ == "__main__":
    sys.exit(main())