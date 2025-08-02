
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import logging

logger = logging.getLogger(__name__)

def create_scheduler(optimizer, config, train_dataloader_len=None, accelerator=None):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
        train_dataloader_len: Length of the training data loader (used to calculate total steps)
        accelerator: Accelerator instance (for logging)
        
    Returns:
        scheduler: Learning rate scheduler instance
        is_epoch_based: Boolean indicating whether the scheduler is epoch-based (e.g., ReduceLROnPlateau) rather than step-based
    """
    # Get learning rate scheduler type from config
    lr_scheduler_type = config.get("training", {}).get("lr_scheduler_type", "linear")
    
    # Calculate total training steps (if needed)
    if train_dataloader_len:
        num_epochs = config.get("training", {}).get("num_epochs", 5)
        total_steps = train_dataloader_len * num_epochs
        warmup_ratio = config.get("training", {}).get("warmup_ratio", 0.01)
        warmup_steps = int(total_steps * warmup_ratio)
        
        if accelerator and accelerator.is_main_process:
            logger.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Flag indicating whether it is epoch-based (default is False, meaning step-based)
    is_epoch_based = False
    
    # Create different schedulers based on type
    if lr_scheduler_type == "linear":
        if not train_dataloader_len:
            raise ValueError("Using linear scheduler requires train_dataloader_len parameter")
            
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        if accelerator and accelerator.is_main_process:
            logger.info(f"Creating linear warmup decay scheduler (linear)")
    
    elif lr_scheduler_type == "cosine":
        if not train_dataloader_len:
            raise ValueError("Using cosine scheduler requires train_dataloader_len parameter")
            
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        if accelerator and accelerator.is_main_process:
            logger.info(f"Creating cosine warmup decay scheduler (cosine)")
    
    elif lr_scheduler_type == "plateau":
        # Get configuration parameters
        patience = config.get("training", {}).get("scheduler_patience", 2)
        factor = config.get("training", {}).get("scheduler_factor", 0.5)
        min_lr = config.get("training", {}).get("scheduler_min_lr", 1e-6)
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',           # Monitor metric should be minimized
            factor=factor,        # Learning rate decay factor
            patience=patience,    # Number of epochs to wait for improvement
            min_lr=min_lr         # Minimum learning rate
        )
        # Mark as epoch-based scheduler
        is_epoch_based = True
        
        if accelerator and accelerator.is_main_process:
            logger.info(f"Creating ReduceLROnPlateau scheduler (patience={patience}, factor={factor})")
    
    elif lr_scheduler_type == "constant":
        # Constant learning rate, no scheduling
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
        
        if accelerator and accelerator.is_main_process:
            logger.info(f"Creating constant learning rate scheduler (constant)")
    
    else:
        raise ValueError(f"Unsupported learning rate scheduler type: {lr_scheduler_type}")
    
    return scheduler, is_epoch_based 