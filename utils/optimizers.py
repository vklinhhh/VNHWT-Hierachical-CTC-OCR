# utils/optimizers.py
import torch.optim as optim
import logging
logger = logging.getLogger(__name__)

def create_optimizer(model, learning_rate, weight_decay=0.01, discriminative_lr=False, encoder_lr_factor=0.1):
    """
    Create optimizer with optional discriminative learning rates for encoder vs rest.

    Args:
        model: The model to optimize.
        learning_rate: Base learning rate for the decoder/new parts.
        weight_decay: Weight decay factor.
        discriminative_lr: Whether to use different LRs for encoder vs rest.
        encoder_lr_factor: Factor to multiply base LR by for the encoder (e.g., 0.1).

    Returns:
        Configured AdamW optimizer.
    """
    if discriminative_lr:
        logger.info(f"Using discriminative learning rate: Encoder LR = {learning_rate * encoder_lr_factor}, Decoder/Other LR = {learning_rate}")
        # Group parameters: vision_encoder vs everything else
        encoder_params = []
        decoder_params = [] # Includes decoder, lm_head, adaptive_layer

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if name.startswith('vision_encoder.'):
                encoder_params.append(param)
            else: # Includes decoder, lm_head, adaptive_layer
                decoder_params.append(param)

        if not encoder_params:
             logger.warning("Discriminative LR enabled, but no parameters found starting with 'vision_encoder.'")
        if not decoder_params:
             logger.warning("Discriminative LR enabled, but no parameters found for decoder/other parts.")

        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': learning_rate * encoder_lr_factor},
            {'params': decoder_params, 'lr': learning_rate}
        ], weight_decay=weight_decay)

    else:
        logger.info(f"Using single learning rate: {learning_rate}")
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), # Only optimize params that require grad
            lr=learning_rate,
            weight_decay=weight_decay
        )

    return optimizer