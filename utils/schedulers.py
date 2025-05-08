# utils/schedulers.py
import math
import logging
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)

class CosineWarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warm-up and cosine annealing.
    Inherits from _LRScheduler for better integration.
    """
    def __init__(self, optimizer, warmup_steps, max_steps, eta_min=0, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of linear warm-up steps.
            max_steps (int): Total number of steps for the cycle (warmup + decay).
            eta_min (float): Minimum learning rate factor (relative to base LR). Default 0.
            last_epoch (int): The index of the last epoch. Default -1.
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max(1, max_steps) # Ensure max_steps is at least 1
        self.eta_min = eta_min
        # Ensure warmup_steps is not greater than max_steps
        if self.warmup_steps > self.max_steps:
             logger.warning(f"warmup_steps ({self.warmup_steps}) > max_steps ({self.max_steps}). Setting warmup_steps = max_steps.")
             self.warmup_steps = self.max_steps

        super().__init__(optimizer, last_epoch)
        logger.info(f"CosineWarmupScheduler initialized: warmup={warmup_steps}, max_steps={max_steps}, eta_min={eta_min}")


    def get_lr(self):
        """Compute learning rate using chainable form of lr transformation"""
        # self.last_epoch starts at -1, is incremented by step() before get_lr() is called.
        # So, current_step = self.last_epoch + 1. Let's use self._step_count which is more direct.
        current_step = self.last_epoch + 1 # Correct way to get current step in _LRScheduler

        if current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = float(current_step) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            # Clamp progress to [0, 1]
            progress = min(progress, 1.0)
            # Cosine annealing formula: eta_min + 0.5 * (1 - eta_min) * (1 + cos(pi * progress))
            lr_scale = self.eta_min + 0.5 * (1.0 - self.eta_min) * (1.0 + math.cos(math.pi * progress))

        # Return list of LRs for each param group
        return [base_lr * lr_scale for base_lr in self.base_lrs]

    # state_dict and load_state_dict are handled by the parent _LRScheduler class.
    # We just need to ensure our parameters (warmup_steps, max_steps, eta_min) are saved/loaded.
    # The parent class saves necessary state like base_lrs, last_epoch.

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`."""
        state = super().state_dict()
        state['warmup_steps'] = self.warmup_steps
        state['max_steps'] = self.max_steps
        state['eta_min'] = self.eta_min
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state."""
        self.warmup_steps = state_dict.pop('warmup_steps')
        self.max_steps = state_dict.pop('max_steps')
        self.eta_min = state_dict.pop('eta_min')
        super().load_state_dict(state_dict)


class CosineWarmupWithPlateauScheduler(object):
    """
    Combined scheduler that uses cosine warmup followed by ReduceLROnPlateau.
    This scheduler starts with a warmup period and cosine decay, then switches
    to ReduceLROnPlateau mode after a specified number of steps.
    """
    def __init__(self, optimizer, warmup_steps, max_steps, plateau_patience=5,
                 plateau_factor=0.5, plateau_min_lr=1e-6, plateau_metric='val_cer',
                 plateau_mode='min', verbose=True):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of linear warm-up steps.
            max_steps (int): Number of steps for initial cosine schedule.
            plateau_patience (int): Number of epochs with no improvement after which LR will be reduced.
            plateau_factor (float): Factor by which LR will be reduced.
            plateau_min_lr (float): Minimum LR value.
            plateau_metric (str): Metric to monitor for plateau scheduling.
            plateau_mode (str): 'min' or 'max' depending on whether lower or higher metric is better.
            verbose (bool): If True, prints a message for each LR update.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.plateau_metric = plateau_metric
        self.cosine_scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps, max_steps
        )
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=plateau_mode,
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=plateau_min_lr,
            verbose=verbose
        )
        self.current_step = 0
        self.using_plateau = False
        
        logger.info(f"CosineWarmupWithPlateauScheduler initialized: "
                   f"warmup={warmup_steps}, max_steps={max_steps}, "
                   f"plateau_patience={plateau_patience}, plateau_factor={plateau_factor}, "
                   f"plateau_metric={plateau_metric}")
    
    def step(self, metric=None):
        """
        Step function that updates learning rate.
        During warmup/cosine phase, use step count.
        After that, use validation metric for plateau scheduling.
        
        Args:
            metric (float, optional): Validation metric value. Required for plateau phase.
        """
        if self.current_step < self.max_steps:
            # Still in cosine phase
            self.cosine_scheduler.step()
            self.current_step += 1
            return
        
        # Switch to plateau mode if not already
        if not self.using_plateau:
            logger.info(f"Switching to ReduceLROnPlateau scheduler after {self.current_step} steps.")
            self.using_plateau = True
        
        # In plateau phase, require metric
        if metric is None:
            logger.warning(f"Metric value required for ReduceLROnPlateau but none provided. Skipping step.")
            return
        
        # Update plateau scheduler
        self.plateau_scheduler.step(metric)
        self.current_step += 1
    
    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        return {
            'cosine_state': self.cosine_scheduler.state_dict(),
            'plateau_state': self.plateau_scheduler.state_dict(),
            'current_step': self.current_step,
            'using_plateau': self.using_plateau
        }
    
    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self.cosine_scheduler.load_state_dict(state_dict['cosine_state'])
        self.plateau_scheduler.load_state_dict(state_dict['plateau_state'])
        self.current_step = state_dict['current_step']
        self.using_plateau = state_dict['using_plateau']