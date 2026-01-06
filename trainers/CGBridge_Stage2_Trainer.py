"""
Training script - CGBridge Stage 2: Cross-Modal Alignment.
For training alignment between code graphs and text
"""
import os
import logging
import argparse
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from accelerate import Accelerator

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts import utils

from models.CGBridge_Stage2 import CodeGraphTextDataset, CGBridgeStage2

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="train CGBridge Stage 2")
    parser.add_argument(
        "--config", type=str, default="@CGBridge_Stage2.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint for resuming training"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )
    parser.add_argument("--new_lr", type=bool, help="Whether to use a new learning rate")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration file"""
    if config_path.startswith('@'):
        # Relative path, based on configs directory
        config_path = os.path.join(
            utils.project_dir(), 
            'configs',
            config_path[1:]
        )
    
    logger.info(f"Loading configuration file: {config_path}")
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path}")
    
    return config

def save_checkpoint(accelerator, model, optimizer, scheduler, epoch, step, output_dir, is_best=False, val_metrics=None, best_val_loss=None):
    """Save model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    
    latest_path = os.path.join(output_dir, "checkpoint_latest")
    accelerator.save_state(latest_path)
    
    training_state = {
        "epoch": epoch,
        "step": step,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss
    }
    
    training_state_path = os.path.join(latest_path, "training_state.pt")
    torch.save(training_state, training_state_path)

    if accelerator.is_main_process:
        logger.info(f"Saved latest checkpoint to {latest_path}")
    
    # If it is the current best, save as best_model
    if is_best:
        best_model_path = os.path.join(output_dir, "best_model")
        accelerator.save_state(best_model_path)
        
        # Also save training state
        best_training_state_path = os.path.join(best_model_path, "training_state.pt")
        torch.save(training_state, best_training_state_path)
        
        if accelerator.is_main_process:
            logger.info(f"Saved current best model to {best_model_path}")

def load_checkpoint(accelerator, checkpoint_path):
    """Load model checkpoint"""
    if accelerator.is_main_process:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Use Accelerate's state recovery feature
    accelerator.load_state(checkpoint_path)
    
    # Load training state information
    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")
        epoch = training_state.get("epoch", 0)
        step = training_state.get("step", 0)
        best_val_loss = training_state.get("best_val_loss", float('inf'))
        
        if accelerator.is_main_process:
            logger.info(f"Restored training state: Epoch {epoch}, Step {step}, Best validation loss {best_val_loss}")
        
        return epoch, step, best_val_loss
    else:
        if accelerator.is_main_process:
            logger.warning(f"Training state file not found: {training_state_path}, will start training from scratch")
        return 0, 0, float('inf')

def create_dataloaders(config):
    """Create training and validation data loaders"""
    # Training dataset
    train_dataset = CodeGraphTextDataset(
        config["data"]["train_path"],
        max_length=config['model']['max_txt_len']
    )
    
    # Validation dataset
    valid_dataset = CodeGraphTextDataset(
        config["data"]["valid_path"],
        max_length=config['model']['max_txt_len']
    )
    
    def collate_graph_text(batch):
        """
        Collate function to handle variable-length node embeddings.
        Supports either graph-level vector (shape [dim]) or node sequence (shape [num_nodes, dim]).
        """
        idx_list, graph_list, code_list, code_emb_list = zip(*batch)

        # Determine max sequence length and embedding dim
        seq_lens = []
        emb_dim = None
        for g in graph_list:
            if g.dim() == 1:
                seq_lens.append(1)
                emb_dim = g.size(-1)
            elif g.dim() == 2:
                seq_lens.append(g.size(0))
                emb_dim = g.size(-1)
            else:
                raise ValueError(f"Unsupported graph embedding shape: {g.shape}")
        max_len = max(seq_lens)

        # Prepare padded tensors
        graph_embeds = torch.zeros(len(batch), max_len, emb_dim, dtype=graph_list[0].dtype)
        graph_atts = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, g in enumerate(graph_list):
            if g.dim() == 1:
                g = g.unsqueeze(0)
            cur_len = g.size(0)
            graph_embeds[i, :cur_len] = g
            graph_atts[i, :cur_len] = 1

        idx_tensor = torch.tensor(idx_list, dtype=torch.long)

        return (idx_tensor, (graph_embeds, graph_atts), list(code_list), list(code_emb_list))
    
    # Create data loaders (no need to explicitly set DistributedSampler, Accelerate will handle it)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,  # Accelerate will handle correct shuffle behavior
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_graph_text,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        collate_fn=collate_graph_text,
    )
    
    return train_loader, valid_loader

def train_epoch(
    model, 
    data_loader, 
    optimizer, 
    scheduler, 
    accelerator,
    epoch, 
    config, 
    start_step=0,
):
    """Single training epoch"""
    model.train()
    
    # log_freq = config.get("log_freq", 100)
    log_freq = config.get('trainer', {}).get('log_freq', 100)
    loss_sum = 0
    loss_count = 0
    
    # Add cumulative loss tracking for the entire epoch
    epoch_loss_sum = 0
    epoch_gtc_loss_sum = 0
    epoch_gtm_loss_sum = 0
    epoch_gtg_loss_sum = 0
    epoch_samples = 0
    
    # Use tqdm to wrap the data loader, only display in the main process
    data_iter = enumerate(data_loader)
    if accelerator.is_main_process:
        data_iter = tqdm(
            data_iter, 
            total=len(data_loader),
            desc=f"Epoch {epoch}",
            # ncols=100,
            position=0, 
            leave=True
        )
    
    for step, samples in data_iter:
        # Skip already trained steps (for resuming training)
        if step < start_step:
            continue
        
        idx, graph_embs, text, _ = samples
        new_samples = (idx, graph_embs, text, None)
        
        # Forward pass
        outputs = model(new_samples)
        loss = outputs["loss"]
        
        # Backward pass - using Accelerate optimization
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        
        # Record loss
        loss_sum += loss.item()
        loss_count += 1
        
        # Accumulate loss for the entire epoch
        epoch_loss_sum += loss.item()
        epoch_gtc_loss_sum += outputs['loss_gtc'].item()
        epoch_gtm_loss_sum += outputs['loss_gtm'].item()
        epoch_gtg_loss_sum += outputs['loss_gtg'].item()
        epoch_samples += 1
        
        # Update tqdm progress bar to show current loss
        if accelerator.is_main_process:
            current_loss = loss.item()
            current_gtc = outputs['loss_gtc'].item()
            current_gtm = outputs['loss_gtm'].item()
            current_gtg = outputs['loss_gtg'].item()
            lr = optimizer.param_groups[0]["lr"]
            
            # Update progress bar postfix
            if isinstance(data_iter, tqdm):
                data_iter.set_postfix(
                    loss=f"{current_loss:.4f}",
                    gtc=f"{current_gtc:.4f}",
                    gtm=f"{current_gtm:.4f}",
                    gtg=f"{current_gtg:.4f}",
                    lr=f"{lr:.4e}"
                )
            
        # Periodically log training progress
        if (step + 1) % log_freq == 0 and accelerator.is_main_process:
            avg_loss = loss_sum / loss_count
            lr = optimizer.param_groups[0]["lr"]
            
            logger.info(
                f"Epoch: {epoch}, Step: {step+1}/{len(data_loader)}, "
                f"Loss: {avg_loss:.4f}, GTC: {outputs['loss_gtc'].item():.4f}, "
                f"GTM: {outputs['loss_gtm'].item():.4f}, GTG: {outputs['loss_gtg'].item():.4f}, "
                f"LR: {lr:.4e}"
            )
            loss_sum = 0
            loss_count = 0
    
    # Calculate average loss for the entire epoch
    if epoch_samples > 0 and accelerator.is_main_process:
        epoch_avg_loss = epoch_loss_sum / epoch_samples
        epoch_avg_gtc_loss = epoch_gtc_loss_sum / epoch_samples
        epoch_avg_gtm_loss = epoch_gtm_loss_sum / epoch_samples
        epoch_avg_gtg_loss = epoch_gtg_loss_sum / epoch_samples
        
        logger.info(
            f"Epoch {epoch} training complete - Average loss: Loss: {epoch_avg_loss:.4f}, "
            f"GTC: {epoch_avg_gtc_loss:.4f}, GTM: {epoch_avg_gtm_loss:.4f}, "
            f"GTG: {epoch_avg_gtg_loss:.4f}"
        )
    
    return step + 1

def validate(model, data_loader, accelerator, config):
    """Validate model performance"""
    model.eval()
    
    total_loss = 0
    total_gtc_loss = 0
    total_gtm_loss = 0
    total_gtg_loss = 0
    step_count = 0
    
    # Use tqdm to wrap the data loader, only display in the main process
    if accelerator.is_main_process:
        val_iter = tqdm(
            data_loader, 
            desc="Validation",
            # ncols=100,
            position=0, 
            leave=True
        )
    else:
        val_iter = data_loader
    
    with torch.no_grad():
        for samples in val_iter:
            idx, graph_embs, text, _ = samples
            
            new_samples = (idx, graph_embs, text, None)
            
            # Forward pass
            outputs = model(new_samples)
            
            # Record loss
            current_loss = outputs["loss"].item()
            current_gtc = outputs["loss_gtc"].item()
            current_gtm = outputs["loss_gtm"].item()
            current_gtg = outputs["loss_gtg"].item()
            
            total_loss += current_loss
            total_gtc_loss += current_gtc
            total_gtm_loss += current_gtm
            total_gtg_loss += current_gtg
            step_count += 1
            
            # Update progress bar to show current loss
            if accelerator.is_main_process and isinstance(val_iter, tqdm):
                val_iter.set_postfix(
                    loss=f"{current_loss:.4f}",
                    gtc=f"{current_gtc:.4f}",
                    gtm=f"{current_gtm:.4f}",
                    gtg=f"{current_gtg:.2e}"
                )
    
    # Gather results from all processes
    gathered_metrics = accelerator.gather(torch.tensor([total_loss, total_gtc_loss, total_gtm_loss, total_gtg_loss, step_count], device=accelerator.device))
    
    # Calculate average loss
    if accelerator.is_main_process:
        gathered_metrics = gathered_metrics.view(-1, 5)
        total_loss = gathered_metrics[:, 0].sum().item()
        total_gtc_loss = gathered_metrics[:, 1].sum().item()
        total_gtm_loss = gathered_metrics[:, 2].sum().item()
        total_gtg_loss = gathered_metrics[:, 3].sum().item()
        step_count = gathered_metrics[:, 4].sum().item()
        
        avg_loss = total_loss / step_count
        avg_gtc_loss = total_gtc_loss / step_count
        avg_gtm_loss = total_gtm_loss / step_count
        avg_gtg_loss = total_gtg_loss / step_count
        
        logger.info(
            f"Validation results - Loss: {avg_loss:.4f}, GTC: {avg_gtc_loss:.4f}, "
            f"GTM: {avg_gtm_loss:.4f}, GTG: {avg_gtg_loss:.4f}"
        )
        
        metrics = {
            "loss": avg_loss,
            "loss_gtc": avg_gtc_loss,
            "loss_gtm": avg_gtm_loss,
            "loss_gtg": avg_gtg_loss,
        }
    else:
        # Non-main process returns dummy values, which will not be used
        metrics = {
            "loss": 0.0,
            "loss_gtc": 0.0,
            "loss_gtm": 0.0,
            "loss_gtg": 0.0,
        }
    
    # Synchronize all processes
    accelerator.wait_for_everyone()
    return metrics

def setup_logger(output_dir, log_name="training"):
    """Set up logger and add file output"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file name with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{log_name}_{timestamp}.log")
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Set format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if root_logger.handlers:
        root_logger.handlers = []
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logs will be saved to: {log_file}")
    return logger

def main():
    """Main training function"""
    args = parse_args()
    # Load configuration
    config = load_config(args.config)
    output_dir = config["output"]["save_dir"]
    
    # Initialize Accelerator, set find_unused_parameters=True
    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    print("LOCAL_RANK", accelerator.local_process_index, "→ DEVICE", accelerator.device)
    
    # Create output directory and set up logging
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        # Set up logging to file
        setup_logger(output_dir)
        logger.info(f"Accelerate initialized successfully, number of processes: {accelerator.num_processes}, device: {accelerator.device}")
        
        # Log ablation study settings
        trainer_config = config.get("trainer", {})
        objectives_config = trainer_config.get("objectives", {})
        enable_gtc = objectives_config.get('enable_gtc', True)
        enable_gtm = objectives_config.get('enable_gtm', True)
        enable_gtg = objectives_config.get('enable_gtg', True)
        
        logger.info("="*50)
        logger.info("Ablation study configuration:")
        logger.info(f"  - GTC (Contrastive Learning): {enable_gtc} (always enabled)")
        logger.info(f"  - GTM (Matching Learning): {enable_gtm}")
        logger.info(f"  - GTG (Language Modeling):   {enable_gtg}")
        logger.info("="*50)
        
        # Remove code that automatically adds ablation identifiers, directly use output directory from config
        
        # Save configuration file
        config_save_path = os.path.join(output_dir, "config.yaml")
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)
    
    # Create data loaders
    train_loader, valid_loader = create_dataloaders(config)
    
    # Update device configuration
    model_config = config.copy()
    model_config["device"] = "cuda" if accelerator.device.type == "cuda" else "cpu"
    
    # Create model
    model = CGBridgeStage2(model_config)
    
    # Set optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config.get("trainer", {}).get("lr", 1e-4)),
        weight_decay=float(config.get("trainer", {}).get("weight_decay", 0.01)),
    )
    # Calculate training steps
    num_epochs = config.get("trainer", {}).get("num_epochs", 10)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * config.get("trainer", {}).get("warmup_ratio", 0.01))
    if accelerator.is_main_process:
        logger.info(f"Training steps: {total_steps}, Warmup steps: {warmup_steps}")

    # Set learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Prepare model, optimizer, and data loaders with Accelerator
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
            
    # Variables for resuming training
    start_epoch = 0
    start_step = 0
    best_val_loss = float('inf')
    
    # If a checkpoint is provided, load after prepare
    if args.checkpoint:
        # Check if it's a directory or file path
        checkpoint_path = args.checkpoint
        if not os.path.isdir(checkpoint_path):
            # It might be an old .pt file format, try to convert to new format directory
            if accelerator.is_main_process:
                logger.warning(f"Detected old version checkpoint format: {checkpoint_path}, trying to convert")
            # Here you can add code to convert from old format to new format
            # ...
        
        # Load checkpoint
        start_epoch, start_step, best_val_loss = load_checkpoint(accelerator, checkpoint_path)
        
        # Reset learning rate (new code)
        if args.new_lr:
            new_lr = float(config.get("trainer", {}).get("lr", 1e-4))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            # Reinitialize learning rate scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
            
            # Reprepare scheduler
            scheduler = accelerator.prepare(scheduler)
            
            if accelerator.is_main_process:
                logger.info(f"Reset learning rate to: {new_lr}")

    # Start training
    if accelerator.is_main_process:
        logger.info(f"Starting training (Total epochs: {num_epochs}, Starting epoch: {start_epoch})")
    
    # Initialize early stopping related variables
    patience = config.get("trainer", {}).get("patience", 3)  # Default to 3 epochs without improvement for early stopping
    patience_counter = 0
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Train one epoch
        step = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            epoch=epoch,
            config=config,
            start_step=(start_step if epoch == start_epoch else 0),
        )
        
        # Clear start_step so that subsequent epochs start from scratch
        start_step = 0
        
        # Validate model
        model.eval()
        val_metrics = validate(model, valid_loader, accelerator, config)

        # Main process handles early stopping and saving
        if accelerator.is_main_process:
            # Check if it's the best model
            val_loss = val_metrics["loss"]
            is_best = val_loss < best_val_loss
            
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")
            
            # Save checkpoint (latest and best if applicable)
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,  # Save starting point for the next epoch
                step=step,
                output_dir=output_dir,
                is_best=is_best,
                val_metrics=val_metrics,
                best_val_loss=best_val_loss
            )
            
            # Check if early stopping should be triggered
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs, best validation loss: {best_val_loss:.4f}")
                break
        
        # Synchronize all processes
        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info(f"Training complete! Final model saved in {output_dir}, best model saved as best_model")

    accelerator.wait_for_everyone()
    
        

if __name__ == "__main__":
    main() 
    


'''


accelerate launch --num_processes 8 CGBridge_Stage2_Trainer.py --config @stage2_configs/stage2-gt-ACD-2-unixcoder-32q.yaml

'''
