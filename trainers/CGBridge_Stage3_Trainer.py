"""
Training script - CGBridge Stage 3: Instruction-Based Task Adaptation.
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
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import shutil

import numpy as np
from transformers import BertTokenizer

from rouge_score import rouge_scorer

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts import utils
from models.CGBridge_Stage3 import CGBridgeStage3, CodeSummaryDataset, CodeTranslateDataset
from models.schedulers import create_scheduler

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train CGBridge Stage3")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the configuration file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to the checkpoint to resume training"
    )
    parser.add_argument("--new_lr", action="store_true", help="Whether to use a new learning rate")
    parser.add_argument("--new_scheduler", action="store_true", help="Whether to use a new scheduler")
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
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    
    # Define paths
    latest_path = os.path.join(output_dir, "checkpoint_latest")
    training_state_path = os.path.join(latest_path, "training_state.pt")
    
    # 1. Always save the latest checkpoint state
    accelerator.save_state(latest_path)
    
    # 2. Save additional training state information to the latest checkpoint directory
    training_state = {
        "epoch": epoch,
        "step": step,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss
    }
    # Only save training state file in the main process, as save_state may only write or handle distributed writes in the main process
    if accelerator.is_main_process:
        torch.save(training_state, training_state_path)
        logger.info(f"Saved latest checkpoint to {latest_path}")

    # 3. If it is the current best, copy the latest checkpoint to the best_model directory
    if is_best and accelerator.is_main_process:
        best_model_path = os.path.join(output_dir, "best_model")
        
        # If the best_model directory already exists, delete it first
        
        if os.path.exists(best_model_path):
            shutil.rmtree(best_model_path)
            
        # Copy the entire latest_path directory to best_model_path
        try:
            shutil.copytree(latest_path, best_model_path)
            logger.info(f"Detected best model, copying latest checkpoint to {best_model_path}")
        except Exception as e:
            logger.error(f"Failed to copy checkpoint directory: {e}")
    # if is_best:
    #     best_model_path = os.path.join(output_dir, "best_model")
    #     accelerator.save_state(best_model_path)
        
    #     # Also save training state
    #     best_training_state_path = os.path.join(best_model_path, "training_state.pt")
    #     torch.save(training_state, best_training_state_path)
        
    #     if accelerator.is_main_process:
    #         logger.info(f"Saved current best model to {best_model_path}")


def load_checkpoint(accelerator, checkpoint_path, skip_scheduler=False):
    """Load model checkpoint"""
    if accelerator.is_main_process:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        if skip_scheduler:
            logger.info("Skipping loading scheduler state")
    
    # Use Accelerate's state recovery feature
    try:
        if skip_scheduler:
            # Save current scheduler state
            temp_schedulers = accelerator._schedulers.copy() if hasattr(accelerator, "_schedulers") else None
            
            # Load all states except the scheduler
            accelerator.load_state(checkpoint_path)
            
            # Restore original scheduler
            if temp_schedulers is not None:
                accelerator._schedulers = temp_schedulers
            
            if accelerator.is_main_process:
                logger.info(f"Checkpoint state loaded (skipped scheduler)")
        else:
            # Normally load all states
            accelerator.load_state(checkpoint_path)
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"Error loading checkpoint: {e}")
            if not skip_scheduler:
                logger.warning("Attempting to reload by skipping scheduler...")
                return load_checkpoint(accelerator, checkpoint_path, skip_scheduler=True)
            else:
                raise e
    
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
            logger.warning(f"Training state file not found: {training_state_path}, starting training from scratch")
        return 0, 0, float('inf')

def create_dataloaders(config):
    """Create training and validation data loaders based on task"""
    # Get task type
    task = config.get("task", "summarization").lower()
    
    # Select dataset class based on task type
    if task == "translation":
        dataset_class = CodeTranslateDataset
        logger.info("Using code translation dataset: CodeTranslateDataset")
    else:  # Default to summarization
        dataset_class = CodeSummaryDataset
        logger.info(f"Using code summarization dataset: CodeSummaryDataset (task={task})")
    
    # Training dataset
    train_dataset = dataset_class(
        csv_path=config["data"]["train_path"],
        max_length=config['model']['max_txt_len']
    )
    
    # Validation dataset
    valid_dataset = dataset_class(
        csv_path=config["data"]["valid_path"],
        max_length=config['model']['max_txt_len']
    )

    def collate_graph_text(batch):
        """
        Collate function to handle variable-length node embeddings.
        Supports either graph-level vector (shape [dim]) or node sequence (shape [num_nodes, dim]).
        """
        idx_list, graph_list, *text_fields = zip(*batch)

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

        graph_embeds = torch.zeros(len(batch), max_len, emb_dim, dtype=graph_list[0].dtype)
        graph_atts = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, g in enumerate(graph_list):
            if g.dim() == 1:
                g = g.unsqueeze(0)
            cur_len = g.size(0)
            graph_embeds[i, :cur_len] = g
            graph_atts[i, :cur_len] = 1

        idx_tensor = torch.tensor(idx_list, dtype=torch.long)

        text_outputs = [list(field) for field in text_fields]
        return (idx_tensor, (graph_embeds, graph_atts), *text_outputs)
    
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
    is_scheduler_epoch_based=False,
):
    """Single training epoch"""
    model.train()
    
    log_freq = config.get('output', {}).get('log_freq', 100)
        
    loss_sum = 0
    loss_count = 0
    
    # Add cumulative loss tracking for the entire epoch
    epoch_loss_sum = 0
    epoch_samples = 0
    
    # Use tqdm to wrap the data loader, only display in the main process
    data_iter = enumerate(data_loader)
    if accelerator.is_main_process:
        data_iter = tqdm(
            data_iter, 
            total=len(data_loader),
            desc=f"Epoch {epoch}",
            position=0, 
            leave=True
        )
    
    for step, samples in data_iter:
        # Skip already trained steps (for resuming training)
        if step < start_step:
            continue
        
        idx, graph_embs, code, code_summary = samples
        
        new_samples = (idx, graph_embs, code, code_summary)
        
        # Forward pass
        outputs = model(new_samples)
        loss = outputs["loss"]
        
        # Backward pass - using Accelerate optimization
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        
        # Only step the scheduler after each training step if it is not epoch-based
        if not is_scheduler_epoch_based:
            scheduler.step()
        
        # Record loss
        loss_sum += loss.item()
        loss_count += 1
        
        # Accumulate the entire epoch's loss
        epoch_loss_sum += loss.item()
        epoch_samples += 1
        
        # Update tqdm progress bar to show current loss
        if accelerator.is_main_process:
            current_loss = loss.item()
            lr = optimizer.param_groups[0]["lr"]
            
            # Update progress bar suffix
            if isinstance(data_iter, tqdm):
                data_iter.set_postfix(
                    loss=f"{current_loss:.2f}",
                    lr=f"{lr:.2e}"
                )
            
        # Periodically log training progress
        if (step + 1) % log_freq == 0 and accelerator.is_main_process:
            avg_loss = loss_sum / loss_count
            lr = optimizer.param_groups[0]["lr"]
            
            logger.info(
                f"Epoch: {epoch}, Step: {step+1}/{len(data_loader)}, "
                f"Loss: {avg_loss:.2f}, LR: {lr:.2e}"
            )
            loss_sum = 0
            loss_count = 0
        
        # Clear cache every n steps
        if step % 50 == 0:
            torch.cuda.empty_cache()
    
    # Calculate average loss for the entire epoch
    if epoch_samples > 0 and accelerator.is_main_process:
        epoch_avg_loss = epoch_loss_sum / epoch_samples
        
        logger.info(
            f"Epoch {epoch} training complete - Average loss: {epoch_avg_loss:.4f}"
        )
    torch.cuda.empty_cache()
    return step + 1

def validate(model, data_loader, accelerator, config):
    """Validate model performance"""
    model.eval()
    
    total_loss = 0
    step_count = 0
    
    # Use tqdm to wrap the data loader, only display in the main process
    if accelerator.is_main_process:
        val_iter = tqdm(
            data_loader, 
            desc="Validation",
            position=0, 
            leave=True
        )
    else:
        val_iter = data_loader
    
    with torch.no_grad():
        for batch_idx, samples in enumerate(val_iter):
            # Move data to device - no need to manually move, Accelerate will handle it
            idx, graph_embs, code, code_summary = samples
            
            # Add NaN/Inf check for input graph_embs (supports tuple (embeds, mask))
            graph_to_check = graph_embs[0] if isinstance(graph_embs, tuple) else graph_embs
            if torch.isnan(graph_to_check).any() or torch.isinf(graph_to_check).any():
                accelerator.print(f"VALIDATE_INPUT_CHECK (Batch {batch_idx}): NaN/Inf found in graph_embs from valid_loader! CSV_Indices (first 5): {idx.tolist()[:5]}")

            # Keep text that cannot be directly moved to the device as a list
            new_samples = (idx, graph_embs, code, code_summary)
            
            # Forward pass
            outputs = model(new_samples) # model is the original model after accelerator.unwrap_model(model)
            
            # Record loss
            current_loss_tensor = outputs["loss"]

            if current_loss_tensor is None:
                accelerator.print(f"VALIDATE_ERROR (Batch {batch_idx}): Loss tensor is None! CSV_Indices (first 5): {idx.tolist()[:5]}")
                current_loss = 0.0 # Assign a neutral value to avoid crashing .item()
            elif torch.isnan(current_loss_tensor).any() or torch.isinf(current_loss_tensor).any():
                accelerator.print(f"VALIDATE_ERROR (Batch {batch_idx}): NaN/Inf loss detected! CSV_Indices (first 5): {idx.tolist()[:5]}")
                # Log information about samples in the batch that caused NaN
                for i in range(len(idx)):
                    # Log only if the specific loss for this sample (if computed per sample) or the whole batch loss is NaN
                    # Since loss is usually batch-wise, this logs all samples in the problematic batch
                    accelerator.print(f"    Problematic sample in NaN batch - CSV_idx: {idx[i].item()}, Code (first 100 chars): '{code[i][:100]}...', Summary (first 100 chars): '{code_summary[i][:100]}...'")
                # Save data from the batch that caused NaN, each process saves its own if needed
                # problematic_batch_data = {
                #     'csv_indices': idx.cpu(), 
                #     'graph_embs': graph_embs.cpu(), 
                #     'code': code, 
                #     'code_summary': code_summary
                # }
                # torch.save(problematic_batch_data, f"nan_validation_batch_rank{accelerator.process_index}_batch{batch_idx}.pt")
                current_loss = 0.0 # Or you can use a very large number to mark, but 0.0 can avoid affecting subsequent average calculations (if other batches are normal)
            else:
                current_loss = current_loss_tensor.item()
            
            total_loss += current_loss
            step_count += 1
            
            
            # Update progress bar to show current loss
            if accelerator.is_main_process and isinstance(val_iter, tqdm):
                # Display the problematic current_loss if it was NaN/Inf before being reset
                display_loss = current_loss_tensor.item() if current_loss_tensor is not None else float('nan')
                val_iter.set_postfix(
                    loss=f"{display_loss:.4f}"
                )
    
    # Gather results from all processes
    gathered_metrics = accelerator.gather(torch.tensor([total_loss, step_count], device=accelerator.device))
    
    # Calculate average loss
    if accelerator.is_main_process:
        # Reorganize the gathered metrics into an array
        gathered_metrics = gathered_metrics.view(-1, 2)
        total_loss = gathered_metrics[:, 0].sum().item()
        step_count = gathered_metrics[:, 1].sum().item()
        
        avg_loss = total_loss / step_count
        
        logger.info(f"Validation results - Loss: {avg_loss:.4f}")
        
        
        metrics = {
            "loss": avg_loss,
        }
    else:
        # Non-main process returns dummy values, which will not be used
        metrics = {
            "loss": 0.0,
        }
    
    # Synchronize all processes
    accelerator.wait_for_everyone()
    return metrics

def setup_logger(output_dir, log_name="training"):
    """Set up logger and add file output"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file name, including timestamp
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
    
    # Clear any existing old handlers
    if root_logger.handlers:
        root_logger.handlers = []
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logs will be saved to: {log_file}")
    return logger

def main():
    """Main training function"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    # Load configuration
    config = load_config(args.config)
    output_dir = config["output"]["save_dir"]
    
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    print("LOCAL_RANK", accelerator.local_process_index, "→ DEVICE", accelerator.device)
    
    
    # Create output directory and set up logging
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        # Set up logging to file
        setup_logger(output_dir)
        logger.info(f"Accelerate initialized successfully, number of processes: {accelerator.num_processes}, device: {accelerator.device}")
        
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
    model = CGBridgeStage3(model_config)
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config.get("training", {}).get("lr", 5e-6)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 0.01)),
    )
    
    
    # Calculate number of training steps
    num_epochs = config.get("training", {}).get("num_epochs", 5)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * config.get("training", {}).get("warmup_ratio", 0.01))
    
    # Use scheduler factory to create learning rate scheduler
    scheduler, is_scheduler_epoch_based = create_scheduler(
        optimizer=optimizer,
        config=config,
        train_dataloader_len=len(train_loader),
        accelerator=accelerator
    )
    
    if accelerator.is_main_process:
        logger.info(f"Using learning rate scheduler type: {config.get('training', {}).get('lr_scheduler_type', 'linear')}")
        logger.info(f"Is it an epoch-based scheduler: {is_scheduler_epoch_based}")
    
    # Prepare model, optimizer, and data loaders with Accelerator
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
            
    # Variables for resuming training
    start_epoch = 0
    start_step = 0
    best_val_loss = float('inf')
    
    # If a checkpoint is provided, load it after prepare
    if args.checkpoint:
        # Check if it is a directory or file path
        checkpoint_path = args.checkpoint
        
        # Determine if we need to skip loading the scheduler
        skip_scheduler = args.new_scheduler
        
        # If only modifying the learning rate but not changing the scheduler type, we can load first and then modify
        if args.new_lr and not args.new_scheduler:
            skip_scheduler = False
        
        # Load checkpoint
        start_epoch, start_step, best_val_loss = load_checkpoint(
            accelerator, 
            checkpoint_path,
            skip_scheduler=skip_scheduler
        )
        
        # Handle scheduler and learning rate
        if args.new_scheduler:
            # Create a brand new scheduler (will apply the learning rate from the configuration)
            if accelerator.is_main_process:
                logger.info(f"Creating new scheduler, type: {config.get('training', {}).get('lr_scheduler_type', 'linear')}")
            
            scheduler, is_scheduler_epoch_based = create_scheduler(
                optimizer=optimizer,
                config=config,
                train_dataloader_len=len(train_loader),
                accelerator=accelerator
            )
            scheduler = accelerator.prepare(scheduler)
        
        # Only modify learning rate separately if not creating a new scheduler
        elif args.new_lr:
            new_lr = float(config.get("training", {}).get("lr", 5e-6))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            if accelerator.is_main_process:
                logger.info(f"Only modified base learning rate to: {new_lr}, retaining original scheduler")

    # Start training
    if accelerator.is_main_process:
        logger.info(f"Starting training (Total epochs: {num_epochs}, Starting epoch: {start_epoch})")
    
    # Initialize early stopping related variables
    patience = config.get("training", {}).get("patience", 3)  # Default to early stopping after 3 epochs without improvement
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
            is_scheduler_epoch_based=is_scheduler_epoch_based,
        )
        
        # Clear start_step so that subsequent epochs start from scratch
        start_step = 0
        
        # Validate model
        model.eval()
        val_metrics = validate(model, valid_loader, accelerator, config)

        # If it is an epoch-based scheduler (like ReduceLROnPlateau), step after validation
        if is_scheduler_epoch_based:
            # Step the scheduler in the main process
            if accelerator.is_main_process:
                scheduler.step(val_metrics["loss"])
            # Synchronize all processes to ensure scheduler state consistency
            accelerator.wait_for_everyone()

        # Main process handles early stopping and saving
        if accelerator.is_main_process:
            should_stop = False
            # Check if it is the best model
            val_loss = val_metrics["loss"]
            is_best = val_loss < best_val_loss

            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience count: {patience_counter}/{patience}")
            # Check if early stopping should be triggered
            if patience_counter >= patience:
                should_stop = True
                logger.info(f"Early stopping triggered after {epoch+1} epochs, best validation loss: {best_val_loss:.4f}")
        else:
            should_stop = False
            is_best = False

        # Save checkpoint (latest and best if applicable)
        if accelerator.is_main_process:
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

        # Now 'should_stop' is defined for all processes
        stop_tensor = torch.tensor(int(should_stop), device=accelerator.device)
        stop_list = accelerator.gather(stop_tensor)
        if stop_list.max().item() == 1:
            # Early stopping logic remains unchanged
            if accelerator.is_main_process:
                 logger.info(f"Early stopping triggered after {epoch+1} epochs, best validation loss: {best_val_loss:.4f}")
            break

        # Synchronize all processes
        accelerator.wait_for_everyone()
    
    # Output training completion information
    if accelerator.is_main_process:
        logger.info(f"Training complete! Final model saved in {output_dir}, best model saved as best_model")

    # Clean up distributed environment 
    accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main() 
    
    
"""
# ================translation================
    
accelerate launch --num_processes=8  \
CGBridge_Stage3_Trainer.py \
--config @stage3_configs/trans-qwencoder1.5b-gt-ACD-2-unixcoder-32q.yaml

# ================summarization===============

accelerate launch --num_processes=8  \
CGBridge_Stage3_Trainer.py \
--config @stage3_configs/summ-qwencoder1.5b-gat-ACD-2-unixcoder-32q.yaml


"""
