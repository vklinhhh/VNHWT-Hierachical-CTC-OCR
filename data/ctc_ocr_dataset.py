# data/ctc_ocr_dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import logging
import numpy as np
import unicodedata

logger = logging.getLogger(__name__)

class CtcOcrDataset(Dataset):
    """
    Dataset for CTC-based OCR model.
    Processes images and converts text labels into character index sequences.
    """
    def __init__(self, hf_dataset, processor, char_to_idx_map, unk_token='[UNK]', ignore_case=False):
        """
        Args:
            hf_dataset: HuggingFace dataset with 'image' and text label (e.g., 'label').
            processor: Feature extractor (e.g., from TrOCRProcessor or AutoProcessor).
            char_to_idx_map (dict): Dictionary mapping characters (including blank) to indices.
                                    Must contain a key for the unknown token.
            unk_token (str): String representing the unknown token (e.g., '[UNK]').
            ignore_case (bool): Whether to convert labels to lowercase.
        """
        self.dataset = hf_dataset
        self.processor = processor
        self.char_to_idx = char_to_idx_map
        self.ignore_case = ignore_case
        self.unk_token = unk_token

        if self.unk_token not in self.char_to_idx:
             raise ValueError(f"Unknown token '{self.unk_token}' not found in char_to_idx_map.")
        self.unk_idx = self.char_to_idx[self.unk_token]
        # CTC Blank token is typically index 0
        self.blank_idx = 0 # Assuming blank is at index 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            example = self.dataset[idx]

            # --- Image Processing ---
            image = example['image']
            try:
                if isinstance(image, str): image = Image.open(image).convert("RGB")
                elif isinstance(image, np.ndarray): image = Image.fromarray(image).convert("RGB")
                elif isinstance(image, Image.Image):
                     if image.mode != 'RGB': image = image.convert('RGB')
                else: raise TypeError(f"Unsupported image type: {type(image)}")

                pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
            except (FileNotFoundError, UnidentifiedImageError, TypeError, ValueError) as img_err:
                logger.warning(f"Skipping sample {idx} due to image processing error: {img_err}")
                return None
            except Exception as e:
                 logger.error(f"Unexpected error processing image for sample {idx}: {e}", exc_info=True)
                 return None

            # --- Label Processing ---
            text_label = example.get('label', example.get('word', example.get('text')))
            if text_label is None or not isinstance(text_label, str):
                logger.warning(f"Skipping sample {idx} due to missing or invalid text label.")
                return None

            if self.ignore_case:
                text_label = text_label.lower()

            # Convert label string to list of character indices for CTC
            # Do NOT add BOS/EOS/PAD tokens here. CTC handles sequences directly.
            # Handle unknown characters.
            label_indices = [self.char_to_idx.get(char, self.unk_idx) for char in text_label]

            # Filter out blank tokens from target labels if necessary?
            # Standard CTC loss expects targets *without* the blank token.
            # label_indices = [idx for idx in label_indices if idx != self.blank_idx] # Usually not needed if vocab is correct

            if not label_indices: # Handle empty labels after processing
                 logger.warning(f"Sample {idx} resulted in empty label indices for text: '{text_label}'. Skipping.")
                 return None


            return {
                "pixel_values": pixel_values,      # [C, H, W]
                "labels": torch.tensor(label_indices, dtype=torch.long), # [TargetSeqLen]
                "text": text_label # Keep original text for potential validation decoding
            }

        except Exception as e:
            logger.error(f"Unexpected error getting item {idx}: {e}", exc_info=True)
            return None