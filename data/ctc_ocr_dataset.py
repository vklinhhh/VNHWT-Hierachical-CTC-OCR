# data/ctc_ocr_dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import logging
import numpy as np
import unicodedata
import albumentations as A
from albumentations.pytorch import ToTensorV2 # For converting to PyTorch tensor
import cv2

logger = logging.getLogger(__name__)

class CtcOcrDataset(Dataset):
    """
    Dataset for CTC-based OCR model.
    Processes images and converts text labels into character index sequences.
    """
    def __init__(self, hf_dataset, processor, char_to_idx_map, unk_token='[UNK]', ignore_case=False,is_training=False):
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
        self.is_training = is_training # Store the flag

        if self.unk_token not in self.char_to_idx:
             raise ValueError(f"Unknown token '{self.unk_token}' not found in char_to_idx_map.")
        self.unk_idx = self.char_to_idx[self.unk_token]
        # CTC Blank token is typically index 0
        self.blank_idx = 0 # Assuming blank is at index 0

        # --- Define Augmentation Pipeline ---
        if self.is_training:
            self.augment_transform = A.Compose([
                # Geometric distortions
                A.Rotate(limit=7, p=0.5, border_mode=cv2.BORDER_REPLICATE), # Slight rotation
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), shear=(-5, 5), p=0.7, mode=cv2.BORDER_REPLICATE), # Scale, translate, shear
                A.Perspective(scale=(0.01, 0.05), p=0.3, pad_mode=cv2.BORDER_REPLICATE), # Slight perspective shifts
                A.ElasticTransform(alpha=1, sigma=15, alpha_affine=15, p=0.3, border_mode=cv2.BORDER_REPLICATE), # Simulates paper warp/wobbly lines
                A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3, border_mode=cv2.BORDER_REPLICATE),

                # Blur and Noise
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=(3, 5), p=1.0)
                ], p=0.4), # Apply one of the blurs
                A.GaussNoise(var_limit=(5.0, 30.0), p=0.4),

                # Brightness, Contrast
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80,120),p=0.3),

                # Simulating print/scan issues
                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
                    A.Downscale(scale_min=0.75, scale_max=0.95, interpolation=cv2.INTER_LINEAR, p=1.0), # Simulates lower resolution
                ], p=0.3),

                # Add more subtle ones if needed, like Sharpen, Emboss, ChannelShuffle, etc.
                # A.CLAHE(p=0.2),
                # A.Sharpen(p=0.2),
                # A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, p=0.2) # Small random occlusions
            ])
            logger.info("Training augmentations enabled.")
        else:
            self.augment_transform = None
            logger.info("Training augmentations disabled (validation/test mode).")

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

            if self.is_training and self.augment_transform:
                            # Convert PIL to NumPy array for Albumentations
                            image_np = np.array(image)
                            try:
                                augmented = self.augment_transform(image=image_np)
                                image_aug_np = augmented['image']
                                # Convert back to PIL Image for Hugging Face processor (if it expects PIL)
                                # OR modify processor to take NumPy, OR use ToTensorV2 from Albumentations
                                image = Image.fromarray(image_aug_np) # For HF processor expecting PIL
                            except Exception as aug_err:
                                logger.warning(f"Augmentation failed for sample {idx}: {aug_err}. Using original image.")
                                
            # This typically normalizes and converts to tensor
            try:
                pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
            except Exception as proc_err:
                logger.warning(f"Hugging Face processor failed for sample {idx}: {proc_err}. Skipping.")
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