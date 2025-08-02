"""
Code summarization script
Reads the trained model and generates summaries on the specified dataset, saving the results as CSV
"""

import os
import sys
import logging
import argparse
import yaml
import torch
import pandas as pd
from tqdm import tqdm
import csv
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import gather_object

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.CGBridge_Stage3 import CGBridgeStage3, CodeTranslateDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate code summaries using the trained model")
    parser.add_argument(
        "--base_path", 
        type=str, 
        help="Base path containing the model and configuration files"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="test", 
        choices=["train", "valid", "test"],
        help="Which dataset to use for generating summaries"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory containing result CSV and intermediate files"
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size during generation"
    )
    parser.add_argument(
        "--no_graph", 
        action="store_true",
        help="When set, do not use graph information, only use LLM to generate code summaries"
    )

    return parser.parse_args()

def create_dataset(config, dataset_type):
    """Create dataset"""
    dataset_path = None
    if dataset_type == "train":
        dataset_path = config['data']['train_path']
    elif dataset_type == "valid":
        dataset_path = config['data']['valid_path']
    elif dataset_type == "test":
        dataset_path = config['data']['test_path']
    
    logger.info(f"Using {dataset_type} dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    return CodeTranslateDataset(
        dataset_path,
        max_length=config['model']['max_txt_len'],
    )

def save_results(results, output_path):
    """Save results to CSV file"""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to CSV file
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'code', 'reference', 'generated'])
        writer.writerows(results)
    
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Build configuration and model paths
    config_path = os.path.join(args.base_path, "config.yaml")
    model_path = os.path.join(args.base_path, "best_model")
    
    # Initialize accelerator, adding find_unused_parameters=True to solve multi-card issues
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    print("LOCAL_RANK", accelerator.local_process_index, "→ DEVICE", accelerator.device)
    
    
    # Output process information
    logger.info(f"Process {accelerator.process_index}/{accelerator.num_processes} initialized on device {accelerator.device}")
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Process {accelerator.process_index} loaded configuration file: {config_path}")
    
    # Create dataset
    dataset = create_dataset(config, args.dataset)
    
    # Use a subset for testing, commenting this line allows processing the entire dataset
    # subset = torch.utils.data.Subset(dataset, range(50))
    
    # Create DataLoader, accelerate will automatically handle distributed
    dataloader = torch.utils.data.DataLoader(
        dataset,
        # subset,
        batch_size=args.batch_size,
        shuffle=False,  # Maintain order during inference
        num_workers=4,  # Use multi-threading to load data
        pin_memory=True
    )
    
    # Prepare dataloader with accelerate
    dataloader = accelerator.prepare(dataloader)
    
    # Initialize model
    logger.info(f"Process {accelerator.process_index} starting to load model...")
    model = CGBridgeStage3(config)
    logger.info(f"Process {accelerator.process_index} model class initialization complete")
    
    # Synchronize all processes to ensure all processes successfully initialized the model before prepare
    accelerator.wait_for_everyone()
    
    # Prepare model (before loading weights)
    model = accelerator.prepare(model)
    logger.info(f"Process {accelerator.process_index} model prepared with accelerator complete")
    
    # Add timeout and error handling
    logger.info(f"Process {accelerator.process_index} starting to load model weights...")
    
    # Use try-except for error handling, ensuring all processes synchronize
    try:
        with accelerator.main_process_first():
            accelerator.load_state(model_path)
        logger.info(f"Process {accelerator.process_index} model weights loaded successfully")
    except Exception as e:
        logger.error(f"Process {accelerator.process_index} failed to load model: {e}")
        # Let all processes know an error occurred
        accelerator.print(f"Process {accelerator.process_index} error: {e}")
        # Safely exit all processes
        if accelerator.is_main_process:
            accelerator.print("Main process is exiting, all processes will end")
        accelerator.wait_for_everyone()
        sys.exit(1)
    
    model.eval()
    
    # Wait for all processes to complete initialization
    accelerator.wait_for_everyone()
    
    # Generate results
    if accelerator.is_main_process:
        logger.info(f"Starting to generate translations for {len(dataset)} samples")
    
    # Collect results
    all_results = []
    
    # Only create tqdm progress bar in the main process
    progress_bar = None
    if accelerator.is_main_process:
        progress_bar = tqdm(dataloader, desc="Generating translations (main process)")
    else:
        # Other processes directly use dataloader
        progress_bar = dataloader

    # Create output directory and subdirectory structure
    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)
    process_dir = os.path.join(result_dir, "process_results")
    os.makedirs(process_dir, exist_ok=True)
    
    result_file = os.path.join(result_dir, f"{args.dataset}_translation.csv")
    
    # Each process's result file
    process_file = os.path.join(process_dir, f"rank_{accelerator.process_index}.csv")
    
    # Each process writes its own file header
    with open(process_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'code', 'reference', 'generated'])
    
    # Each process writes the path of the completion flag file
    complete_flag_file = os.path.join(process_dir, f"rank_{accelerator.process_index}.complete")
    
    # To prevent leftover completion flags from previous runs, delete first
    if os.path.exists(complete_flag_file):
        os.remove(complete_flag_file)
    
    # Process data
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                idx, graph_embs, code_text, references = batch
                
                batch_graph_embs = graph_embs.to(accelerator.device)
                batch_code_text = code_text
                batch_references = references
                
                # Generate translations
                gen_config = {
                    'max_new_tokens': config['generation']['max_new_tokens'],
                    'min_new_tokens': config['generation']['min_new_tokens'],
                    'temperature': config['generation']['temperature'],
                    'repetition_penalty': config['generation']['repetition_penalty'],
                }
                
                # Decide whether to use graph information based on parameters
                if args.no_graph:
                    if hasattr(model, 'module'):
                        generated_translations = model.module.generate_llm_only(
                            batch_code_text,
                            gen_config
                        )
                    else:
                        generated_translations = model.generate_llm_only(
                            batch_code_text,
                            gen_config
                        )
                else:
                    if hasattr(model, 'module'):
                        generated_translations = model.module.generate(
                            (batch_graph_embs, batch_code_text),
                            gen_config
                        )
                    else:
                        generated_translations = model.generate(
                            (batch_graph_embs, batch_code_text),
                            gen_config
                        )
                
                # Write after processing each batch, do not accumulate in memory
                with open(process_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for i, code, ref, gen in zip(idx, batch_code_text, batch_references, generated_translations):
                        writer.writerow([i.item(), code, ref, gen])
                
                # Periodically flush to disk to ensure writing
                if batch_idx % 10 == 0:
                    # Create checkpoint file to record progress
                    with open(os.path.join(process_dir, f"rank_{accelerator.process_index}.checkpoint"), 'a') as f:
                        f.write(f"{batch_idx}\n")
                
            except Exception as e:
                logger.error(f"Process {accelerator.process_index} encountered an error while processing batch {batch_idx}: {e}")
                # Log error information for debugging
                with open(os.path.join(process_dir, f"rank_{accelerator.process_index}.error"), 'a') as f:
                    f.write(f"Batch {batch_idx}: {str(e)}\n")
                continue
            finally:
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
    
    # Write completion flag
    with open(complete_flag_file, 'w') as f:
        f.write("done")
    
    # Wait for all processes to complete writing - remove timeout parameter
    try:
        accelerator.wait_for_everyone()
        logger.info(f"Process {accelerator.process_index} all processes synchronized complete")
    except Exception as e:
        logger.warning(f"Process {accelerator.process_index} encountered an error while waiting for other processes: {e}")
    
    # Main process merges results
    if accelerator.is_main_process:
        logger.info("Main process starting to merge results from all processes...")
        unique_results = {}
        total_processed = 0
        
        for i in range(accelerator.num_processes):
            process_file = os.path.join(process_dir, f"rank_{i}.csv")
            complete_file = os.path.join(process_dir, f"rank_{i}.complete")
            
            if os.path.exists(process_file) and os.path.getsize(process_file) > 0:
                process_count = 0
                with open(process_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if len(row) >= 4:
                            idx = int(row[0])
                            unique_results[idx] = row
                            process_count += 1
                
                logger.info(f"Process {i} processed {process_count} samples, completion flag: {os.path.exists(complete_file)}")
                total_processed += process_count
            else:
                logger.warning(f"Result file for process {i} does not exist or is empty: {process_file}")
        
        logger.info(f"Total samples before merging: {total_processed}, unique samples after merging: {len(unique_results)}")
        
        # Check for missing sample indices
        expected_indices = set(range(len(dataset)))
        actual_indices = set(unique_results.keys())
        missing_indices = expected_indices - actual_indices
        
        if missing_indices:
            logger.warning(f"Found {len(missing_indices)} missing samples, indices: {sorted(list(missing_indices))[:10]}...")
        
        # Sort and write to final file
        sorted_results = [unique_results[idx] for idx in sorted(unique_results.keys())]
        with open(result_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'code', 'reference', 'generated'])
            writer.writerows(sorted_results)
        
        logger.info(f"Final results saved to {result_file}")
        logger.info(f"Original dataset size: {len(dataset)} samples")
        logger.info(f"Successfully processed: {len(sorted_results)} samples")
        logger.info(f"Missing samples: {len(dataset) - len(sorted_results)} samples")
    
    # Wait for the main process to complete file writing
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info("Translation generation complete!")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Destroyed process group, cleaned up resources")


    
if __name__ == "__main__":
    main()


"""
# Multi-card running example
accelerate launch \
    --num_processes=8 \
    --gpu_ids="0,1,2,3,4,5,6,7" \
    --main_process_port=29500 \
    translate.py \
    --base_path /path/to/outputs/stage3/trans-qwencoder1.5b-ACD-2-unixcoder-32q \
    --dataset test \
    --batch_size 16 \
    --output_dir /path/to/results/stage3/trans-qwencoder1.5b-ACD-2-unixcoder-32q



"""