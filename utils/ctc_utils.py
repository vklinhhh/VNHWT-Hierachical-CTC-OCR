# utils/ctc_utils.py
import torch
import itertools
import logging
import unicodedata # For building combined vocab

logger = logging.getLogger(__name__)

# --- CTCDecoder class remains the same ---
class CTCDecoder:
    """Basic CTC Greedy Decoder."""
    def __init__(self, idx_to_char_map, blank_idx=0, output_delimiter=''):
        self.idx_to_char = idx_to_char_map
        self.blank_idx = blank_idx
        self.output_delimiter = output_delimiter

    def __call__(self, logits):
        if logits.ndim != 3: raise ValueError(f"Logits 3D, got {logits.shape}")
        logits = logits.cpu().detach()
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded_batch = []
        for pred_seq in predicted_ids:
            merged = [k for k, _ in itertools.groupby(pred_seq.tolist())]
            cleaned = [k for k in merged if k != self.blank_idx]
            try:
                decoded_elements = [self.idx_to_char.get(idx, '?') for idx in cleaned]
                decoded_string = self.output_delimiter.join(decoded_elements)
            except Exception as e:
                logger.warning(f"Decode map error: {e}. Sequence: {cleaned}")
                decoded_string = "<DECODE_ERROR>"
            decoded_batch.append(decoded_string)
        return decoded_batch


# --- build_ctc_vocab remains the same ---
def build_ctc_vocab(char_list, add_blank=True, add_unk=True, unk_token='[UNK]'):
    vocab = []
    if add_blank: vocab.append('<blank>')
    unique_chars = sorted(list(set(char_list)))
    vocab.extend(unique_chars)
    if add_unk:
        if unk_token not in vocab: vocab.append(unk_token)
        else: logger.warning(f"'{unk_token}' already in char_list.")
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    logger.info(f"Built CTC vocab with {len(vocab)} tokens (Blank: {add_blank}, UNK: {add_unk})")
    return vocab, char_to_idx, idx_to_char


# --- NEW: Function to build COMBINED character vocab ---
def build_combined_vietnamese_charset(include_basic_latin=True, include_digits=True, include_punctuation=True, additional_chars=""):
    """Generates a comprehensive set of combined Vietnamese characters."""
    charset = set()
    vowels = "aeiouy"
    diacritic_codes = [
        "", # No diacritic
        "\u0301", # Acute
        "\u0300", # Grave
        "\u0309", # Hook above
        "\u0303", # Tilde
        "\u0323", # Dot below
    ]
    # Modifiers applicable only to specific vowels
    circumflex = "\u0302" # on a, e, o
    breve = "\u0306"    # on a
    horn = "\u031b"     # on u, o -> ư, ơ

    for base_vowel in vowels:
        for case_func in [str.lower, str.upper]:
            bv = case_func(base_vowel)
            # Base vowel + simple diacritics
            for diac in diacritic_codes:
                try: charset.add(unicodedata.normalize('NFC', bv + diac))
                except: pass
            # Circumflex combinations (a, e, o)
            if base_vowel in "aeo":
                try:
                    bv_circ = unicodedata.normalize('NFC', bv + circumflex)
                    charset.add(bv_circ)
                    for diac in diacritic_codes: # Add simple diacritics on top
                         try: charset.add(unicodedata.normalize('NFC', bv_circ + diac))
                         except: pass
                except: pass
            # Breve combinations (a)
            if base_vowel in "a":
                try:
                    bv_breve = unicodedata.normalize('NFC', bv + breve)
                    charset.add(bv_breve)
                    for diac in diacritic_codes: # Add simple diacritics on top
                         try: charset.add(unicodedata.normalize('NFC', bv_breve + diac))
                         except: pass
                except: pass
            # Horn combinations (u, o) -> ư, ơ
            if base_vowel in "uo":
                try:
                    bv_horn = unicodedata.normalize('NFC', bv + horn)
                    charset.add(bv_horn)
                    for diac in diacritic_codes: # Add simple diacritics on top
                         try: charset.add(unicodedata.normalize('NFC', bv_horn + diac))
                         except: pass
                except: pass

    # Basic Latin Consonants
    if include_basic_latin:
        consonants = "bcdfghjklmnpqrstvwxz"
        for c in consonants: charset.add(c); charset.add(c.upper())

    # Special Vietnamese consonants
    charset.add('đ'); charset.add('Đ')

    # Digits
    if include_digits:
        for d in "0123456789": charset.add(d)

    # Punctuation
    if include_punctuation:
        punct = " .,-_()[]{}:;\"'/\\?!@#$%^&*+=<>|"
        for p in punct: charset.add(p)

    # Additional chars
    for char in additional_chars: charset.add(char)

    return sorted(list(charset))

# Example usage:
# COMBINED_VIETNAMESE_CHARSET = build_combined_vietnamese_charset()
# print(f"Generated {len(COMBINED_VIETNAMESE_CHARSET)} combined characters.")