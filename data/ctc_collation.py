# data/ctc_collation.py
import torch

def ctc_collate_fn(batch):
    """
    Collate function for CTC OCR task.
    Handles filtering None, stacking images, padding labels,
    and creating label_lengths tensor.
    Input lengths (for log_probs) are NOT created here, must be calculated
    after the model's encoder/RNN forward pass based on its output shape.
    """
    # Filter out samples that failed during dataset processing
    batch = [item for item in batch if item is not None]
    if not batch: return None

    # Stack pixel values
    try:
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
    except Exception as e:
        print(f"Error stacking pixel_values: {e}.")
        shapes = [item['pixel_values'].shape for item in batch]
        print(f"Pixel value shapes in failing batch: {shapes}")
        return None # Let DataLoader handle None batch

    # Pad labels (target character indices)
    labels = [item['labels'] for item in batch]
    # Get length of each label BEFORE padding
    label_lengths = torch.tensor([len(lab) for lab in labels], dtype=torch.long)

    # Pad labels - Use 0 (blank index) or another specific pad index if defined.
    # CTC loss often uses blank=0, so padding with 0 might be okay IF blank is handled correctly,
    # but using a dedicated pad index might be safer if vocab includes 0 for a real char.
    # Let's assume blank=0 and we pad with 0. The label_lengths tensor tells CTC loss the real length.
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=0 # Pad with blank index
    )

    # Keep original text if available
    texts = [item.get('text', '') for item in batch]


    return {
        'pixel_values': pixel_values,
        'labels': padded_labels,
        'label_lengths': label_lengths, # Crucial for CTC loss
        'texts': texts # Optional: for validation/debugging
    }