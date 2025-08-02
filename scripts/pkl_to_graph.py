from sympy import pycode
from tqdm import tqdm
import argparse
from python_code_graph import initialize_encoder,code_to_graph
import pandas as pd
import torch
import os
import sys
import numpy as np
import multiprocessing
import shutil
import gc
import time

COLUMN_NAME='src_code'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import  graph_datasets_dir, models_dir,code_datasets_dir

# Helper function to be executed by each process for parallel processing
def worker_fn(worker_id, device, data_slice_df, output_path_slice, model_name_or_path_abs, batch_size, edge_types_list, edge_cache_path_worker, edge_type_to_id_map, current_column_name_val, start_position):
    """
    Processes a slice of the dataset.
    Invoked by multiprocessing.Pool.
    """
    print(f"Worker {worker_id} on device {device} processing {len(data_slice_df)} items. Output to: {output_path_slice}")
    
    # Ensure modules from utils are available if this worker is spawned fresh
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Already done globally
    from python_code_graph import initialize_encoder, code_to_graph # Re-import for safety in some spawn contexts

    tokenizer, code_encoder = initialize_encoder(model_name_or_path_abs, device)
    
    graphs = []
    for position, (_, row) in enumerate(tqdm(data_slice_df.iterrows(), total=len(data_slice_df), desc=f"Worker {worker_id} ({device})", position=worker_id)):
        global_position = start_position + position
        code = row[current_column_name_val]
        try:
            graph = code_to_graph(code, tokenizer, code_encoder, code_from_file=False,
                                  edge_types=edge_types_list, batch_size=batch_size, 
                                  edge_cache_path=edge_cache_path_worker,
                                  edge_type_to_id=edge_type_to_id_map)
            graph.idx = global_position 
            graph.code = code
            graphs.append(graph)
        except Exception as e:
            print(f"Worker {worker_id} (device {device}) failed on item global_position={global_position}: {e}", file=sys.stderr)
            # Optionally, create a graph object with an error field or skip
            # For now, we skip appending if an error occurs during graph creation.

    torch.save({'graphs': graphs}, output_path_slice)
    print(f"Worker {worker_id} ({device}) saved {len(graphs)} graphs to {output_path_slice}")
    return output_path_slice



def parse_args():
    parser = argparse.ArgumentParser(description='.pkl dataset to PyG graph dataset')
    parser.add_argument('--pkl_path', type=str,required=True, help='Path to the .pkl file')
    parser.add_argument('--output_path', type=str,required=True, help='Output path for the PyG graph')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Model name or path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use: cuda:0, cuda:1, cpu, etc.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--edge_types', type=str, default='AST,CFG,DFG', help='Edge types to use: ast, cfg, dfg')
    parser.add_argument('--slice_num', type=int, help='Number of slices')
    parser.add_argument('--slice_idx', type=int, help='Slice index')
    parser.add_argument('--total_size', type=int, help='Total number of samples')
    parser.add_argument('--column_name', type=str, help='')
    
    return parser.parse_args()

def process_all(args):
    """Process the entire dataset (parallel version)"""
    
    # Set multiprocessing start method if using CUDA, do this early
    # Check if any device in args suggests CUDA and if CUDA is available.
    # args.device might be 'cuda:0', 'cuda', or 'cpu'.
    attempt_spawn = False
    if isinstance(args.device, str) and 'cuda' in args.device:
        if torch.cuda.is_available():
            attempt_spawn = True
        else:
            print("CUDA specified but not available. Will use CPU.", file=sys.stderr)
            args.device = 'cpu' # Fallback if user specified cuda but it's not there
    
    if attempt_spawn:
        current_start_method = multiprocessing.get_start_method(allow_none=True)
        if current_start_method != 'spawn':
            try:
                multiprocessing.set_start_method('spawn', force=True)
                print("Set multiprocessing start method to 'spawn' for CUDA compatibility.")
            except RuntimeError as e:
                print(f"Warning: Could not set multiprocessing start method to 'spawn' (Reason: {e}). Using default: {current_start_method}. This might cause issues with CUDA in subprocesses.", file=sys.stderr)

    # Get model name and model_path (absolute)
    model_name = os.path.basename(args.model_name_or_path).split('-')[0]
    model_path_abs = os.path.join(models_dir(), args.model_name_or_path)
    print(f'Using model_name: {model_name}')
    print(f'Using model_path_abs: {model_path_abs}')

    # Set main edge cache path
    main_cache_dir = os.path.join(graph_datasets_dir(), f"enc_by_{model_name}")
    os.makedirs(main_cache_dir, exist_ok=True)
    main_edge_cache_path = os.path.join(main_cache_dir, f"{model_name}_edge_embeddings_cache.pt")
    print(f"Using shared edge cache path: {main_edge_cache_path}")

    # Create final output folder (if not exists)
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    edge_types_list = args.edge_types.split(',')
    edge_type_to_id = {}
    edge_id_to_type = {}
    for i, edge_type_str in enumerate(edge_types_list):
        edge_type_to_id[edge_type_str] = i
        edge_id_to_type[i] = edge_type_str
    
    if args.pkl_path.endswith('.csv'):
        df = pd.read_csv(args.pkl_path)
    elif args.pkl_path.endswith('.pkl'):
        df = pd.read_pickle(args.pkl_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .pkl file.")
    
    # Reset index, ensure using continuous numeric index, avoid interference from original idx field
    df = df.reset_index(drop=True)
    
    total_size = len(df)
    if total_size == 0:
        print("Input .pkl file is empty. Nothing to process.")
        # Create an empty output file consistent with schema
        torch.save({
            'graphs': [],
            'edge_label_to_name': edge_id_to_type
        }, args.output_path)
        print(f"Saved empty graph set to {args.output_path}")
        return

    # Determine number of slices and devices
    devices_to_cycle = []
    if isinstance(args.device, str) and 'cuda' in args.device and torch.cuda.is_available():
        print(f"CUDA_VISIBLE_DEVICES obtained within script: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        if torch.cuda.is_available():
            print(f"torch.cuda.is_available(): True")
            actual_device_count = torch.cuda.device_count()
            print(f"torch.cuda.device_count() returned: {actual_device_count}")
            for i in range(actual_device_count):
                try:
                    print(f"Visible logical device {i}: {torch.cuda.get_device_name(i)}")
                except Exception as e:
                    print(f"Error getting logical device {i} name: {e}")
        else:
            print(f"torch.cuda.is_available(): False")
        available_gpu_count = actual_device_count
        if available_gpu_count == 0:
            print("Warning: CUDA specified, but no GPUs available. Falling back to CPU.", file=sys.stderr)
            devices_to_cycle = ['cpu']
        else:
            # If user specified a specific GPU e.g. cuda:1, try to use only that one if slice_num is 1 or unset
            # For multi-slicing, distribute across available GPUs
            devices_to_cycle = [f'cuda:{i}' for i in range(available_gpu_count)]
    else:
        devices_to_cycle = ['cpu'] # Default to CPU or user specified CPU

    slice_num = args.slice_num
    if slice_num is None or slice_num <= 0:
        slice_num = len(devices_to_cycle) if devices_to_cycle else 1
    slice_num = min(slice_num, total_size) # Cannot have more slices than items
    
    print(f"Targeting {slice_num} slices. Available devices for cycling: {devices_to_cycle}")

    # Create temporary directory for slice results
    temp_output_dir_name = "temp_slices_" + os.path.splitext(os.path.basename(args.output_path))[0]
    temp_output_dir = os.path.join(output_dir, temp_output_dir_name)
    if os.path.exists(temp_output_dir):
        print(f"Cleaning up existing temporary directory: {temp_output_dir}")
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir, exist_ok=True)

    slice_size = (total_size + slice_num - 1) // slice_num
    worker_args_list = []

    # Use the globally set COLUMN_NAME (after main has potentially updated it)
    current_column_name_val = COLUMN_NAME 

    for i in range(slice_num):
        start_idx = i * slice_size
        end_idx = min((i + 1) * slice_size, total_size)
        if start_idx >= end_idx: # Ensure slice is valid
            continue

        data_slice_df = df.iloc[start_idx:end_idx]
        print(data_slice_df.keys())
        print(data_slice_df.head())
        assigned_device = devices_to_cycle[i % len(devices_to_cycle)]
        temp_slice_output_path = os.path.join(temp_output_dir, f"slice_{i}.pt")
        
        worker_args_list.append((
            i, assigned_device, data_slice_df, temp_slice_output_path,
            model_path_abs, args.batch_size, edge_types_list,
            main_edge_cache_path, 
            edge_type_to_id, current_column_name_val, start_idx  # 添加start_idx参数
        ))

    all_graphs = []
    processed_slice_paths = []

    if not worker_args_list:
        print("No data slices to process.")
    elif len(worker_args_list) == 1: # Run in single-process mode for 1 slice
        print("Running in single-process mode as only one slice is generated.")
        s_id, s_device, s_data_slice, s_output_path, s_model_path, s_batch_size, s_edge_types, s_edge_cache, s_edge_map, s_col_name, s_start_idx = worker_args_list[0]
        
        # Re-import for direct call if needed, or ensure worker_fn handles its imports
        # from python_code_graph import initialize_encoder, code_to_graph 
        
        print(f"Processing slice 0 on device {s_device} with {len(s_data_slice)} items.")
        tokenizer_single, code_encoder_single = initialize_encoder(s_model_path, s_device)
        graphs_single_slice = []
        for position, (_, row) in enumerate(tqdm(s_data_slice.iterrows(), total=len(s_data_slice), desc=f"Processing slice 0 ({s_device})")):
            # Use s_start_idx + position as global position index
            global_position = s_start_idx + position
            code = row[s_col_name]
            try:
                graph = code_to_graph(code, tokenizer_single, code_encoder_single, code_from_file=False,
                                      edge_types=s_edge_types, batch_size=s_batch_size, 
                                      edge_cache_path=s_edge_cache, 
                                      edge_type_to_id=s_edge_map)
                graph.idx = global_position  # Use global position instead of DataFrame index
                # graph.code = code
                graphs_single_slice.append(graph)
            except Exception as e:
                print(f"Slice 0 (device {s_device}) failed on item global_position={global_position}: {e}", file=sys.stderr)
        
        torch.save({'graphs': graphs_single_slice}, s_output_path)
        print(f"Slice 0 saved {len(graphs_single_slice)} graphs to {s_output_path}")
        if os.path.exists(s_output_path):
             processed_slice_paths.append(s_output_path)
    else: # Use multiprocessing for more than 1 slice
        print(f"Starting {len(worker_args_list)} workers using multiprocessing.Pool...")
        # Number of processes in the pool should not exceed number of tasks or available device cycle length if that's a constraint
        pool_processes = min(len(worker_args_list), len(devices_to_cycle) if devices_to_cycle else 1, multiprocessing.cpu_count())
        # Or simply use len(worker_args_list) if you want a process per slice up to system limits
        pool_processes = min(len(worker_args_list), multiprocessing.cpu_count()) 
        
        print(f"Using a pool of {pool_processes} processes.")
        with multiprocessing.Pool(processes=pool_processes) as pool:
            processed_slice_paths = pool.starmap(worker_fn, worker_args_list)

    # Collect results
    print("Collecting results from processed slices...")
    for slice_file_path in processed_slice_paths:
        if slice_file_path and os.path.exists(slice_file_path):
            try:
                data = torch.load(slice_file_path, map_location='cpu', weights_only=False)
                if 'graphs' in data:
                    all_graphs.extend(data['graphs'])
                else:
                    print(f"Warning: 'graphs' key not found in {slice_file_path}", file=sys.stderr)
            except Exception as e:
                 print(f"Error loading slice file {slice_file_path}: {e}", file=sys.stderr)
        else:
            print(f"Warning: Slice file {slice_file_path} not found or worker did not return a valid path.", file=sys.stderr)
    
    if all_graphs:
        try:
            # Sort graphs by original index to maintain order if original df order matters
            all_graphs.sort(key=lambda g: g.idx)
        except AttributeError:
            print("Warning: Could not sort graphs by 'idx', perhaps 'idx' attribute is missing on some graph objects.", file=sys.stderr)
            
    # Save combined results
    torch.save({
        'graphs': all_graphs,
        'edge_label_to_name': edge_id_to_type
    }, args.output_path)
    print(f"Saved {len(all_graphs)} graphs to {args.output_path}")

    # Clean up temporary directory
    try:
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
            print(f"Cleaned up temporary directory: {temp_output_dir}")
    except Exception as e:
        print(f"Error cleaning up temporary directory {temp_output_dir}: {e}", file=sys.stderr)

    gc.collect()  # Force garbage collection
    time.sleep(1)


def process_dataset_slice(args):
    """Process a specific slice of the dataset"""
    
    # Get model name
    model_name = os.path.basename(args.model_name_or_path).split('-')[0]
    model_path=os.path.join(models_dir(),args.model_name_or_path)
    print('model_name', model_name)
    
    # Set edge cache path
    cache_dir = os.path.join(graph_datasets_dir(), f"enc_by_{model_name}")
    os.makedirs(cache_dir, exist_ok=True)
    edge_cache_path = os.path.join(cache_dir, f"{model_name}_edge_embeddings_cache.pt")
    
    slice_num = args.slice_num
    slice_size = args.total_size // slice_num
    slice_idx = args.slice_idx
    
    pkl_path = code_datasets_dir()/args.pkl_path
    start_idx = slice_idx * slice_size
    end_idx = (slice_idx + 1) * slice_size
    output_path =  graph_datasets_dir()/args.output_path
    
    edge_types = args.edge_types.split(',')
    edge_type_to_id={}
    edge_id_to_type={}
    for i,edge_type in enumerate(edge_types):
        edge_type_to_id[edge_type] = i
        edge_id_to_type[i] = edge_type
    
    
    # Create output folder (if not exists)
    os.makedirs(output_path, exist_ok=True)
        
    tokenizer, code_encoder = initialize_encoder(model_path, args.device)
    df = pd.read_pickle(pkl_path)
    
    # Reset index, ensure using continuous numeric index
    df = df.reset_index(drop=True)
    
    graphs = []
    
    print(f"Processing slice {slice_idx} of size {slice_size} from {start_idx} to {end_idx}")
    # Only process the specified range of data
    slice_df = df.iloc[start_idx:end_idx]
    del df
    
    for position, (_, row) in enumerate(tqdm(slice_df.iterrows(), total=len(slice_df))):
        # Use start_idx + position as global position index
        global_position = start_idx + position
        # Assume code is in 'code' column
        code = row[COLUMN_NAME]
        graph = code_to_graph(code, tokenizer, code_encoder, code_from_file=False, batch_size=args.batch_size,edge_types=edge_types, edge_cache_path=edge_cache_path, edge_type_to_id=edge_type_to_id)
        # Add code
        # graph.code = code
        # Save global position index instead of DataFrame index
        graph.idx = global_position        
        graphs.append(graph)
        
    
    # Save the result of this slice
    output_file = f"{output_path}/graphs_{slice_idx}_{slice_num}.pt"
    torch.save({
        'graphs': graphs,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'edge_label_to_name': edge_id_to_type   
    }, output_file)
    
    print(f"Saved {len(graphs)} graphs to {output_file}")
    return output_file


if __name__ == '__main__':
    args = parse_args()
    # Update global COLUMN_NAME if user provided it, otherwise use the module-level default.
    if args.column_name is not None:
        COLUMN_NAME = args.column_name
    
    if args.slice_num is not None and args.slice_idx is not None and args.total_size is not None:
        # This condition implies the user wants to run the original single-slice processing directly.
        # Ensure total_size is provided for original slice logic.
        print(f"Processing a specific slice ({args.slice_idx}/{args.slice_num}) as per arguments.")
        process_dataset_slice(args)
    else:
        # Default to processing all, which will now use the parallelized version.
        print("Processing all data. Will use parallel processing if multiple slices are configured/detected.")
        process_all(args)
    

    
"""

============summarization============

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pkl_to_graph.py \
--pkl_path /path/to/code_datasets/test_summarization.csv \
--output_path /path/to/graph_datasets/summ-CD-2-unixcoder/test_summarization.pt \
--model_name_or_path unixcoder-base \
--edge_types CFG,DFG \
--batch_size 1024 \
--device "cuda" --column_name "code" 







============translation============


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pkl_to_graph.py \
--pkl_path /path/to/code_datasets/python2java_test.csv \
--output_path /path/to/graph_datasets/trans-CD-2-unixcoder/python2java_test.pt \
--model_name_or_path unixcoder-base \
--edge_types CFG,DFG \
--batch_size 1024 \
--device "cuda" --column_name "src_code" 




"""
