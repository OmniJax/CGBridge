import json
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pickle  
import pandas as pd
import csv
import numpy as np
from scripts.utils import load_config  
from models.CodeGraphEncoder import CodeGraphEncoder
import torch
import argparse
'''


python produce.py \
--model_path /path/to/outputs/checkpoints/gt_ACD_unixcoder_2_wo-pretrain/cur_best_model.pt \
--config_path /path/to/outputs/checkpoints/gt_ACD_unixcoder_2_wo-pretrain/config.yaml \
--code_data_path /path/to/tasks/summarization/code_datasets/valid_summarization.csv \
--graph_data_path /path/to/tasks/summarization/graph_datasets/valid_summarization_ACD_unixcoder.pt \
--output_path /path/to/tasks/summarization/pair_datasets/sum_ACD_2_unixcoder-wo-pretrain/valid_sum_ACD_2_unixcoder-wo-pretrain.csv \
--batch_size 2048 \
--device cuda:1 \
--column_name code \

'''
def main():
    # Command line argument settings
    parser = argparse.ArgumentParser(description='Generate and store code graph embeddings')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the configuration file')
    parser.add_argument('--code_data_path', type=str, required=True,
                        help='Path to the code dataset to be processed')
    parser.add_argument('--graph_data_path', type=str, required=True,
                        help='Path to the graph dataset to be processed')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output the embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device')
    # Remove --code_encoder parameter
    parser.add_argument('--column_name', type=str, default='code',
                        help='Code column name')
    parser.add_argument('--output_type', type=str, default='node', choices=['graph', 'node'],
                        help='Choose whether to export graph-level embeddings or per-node embeddings')
    parser.add_argument('--use_random_weights', action='store_true',
                   help='Use randomly initialized weights instead of loading the trained model')
    
    args = parser.parse_args()
    
    
    # Create output directory
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creating directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")
    # Load configuration file
    config = load_config(args.config_path)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load code dataset
    print(f"Loading code dataset: {args.code_data_path}")
    try:
        # Choose loading method based on file extension
        if args.code_data_path.endswith('.pkl'):
            with open(args.code_data_path, 'rb') as f:
                code_data = pickle.load(f)
        elif args.code_data_path.endswith('.csv'):
            code_data = pd.read_csv(args.code_data_path)
        else:
            code_data = torch.load(args.code_data_path)
        print(f"Successfully loaded code dataset")
    except Exception as e:
        print(f"Error loading code dataset: {e}")
        return
    
    # Load graph dataset
    print(f"Loading graph dataset: {args.graph_data_path}")
    try:
        graph_data = torch.load(args.graph_data_path, weights_only=False)
        dataset = graph_data['graphs']
        edge_type_map = graph_data.get('edge_label_to_name')
        print(f"Successfully loaded graph dataset with {len(dataset)} graphs")
    except Exception as e:
        print(f"Error loading graph dataset: {e}")
        return
    
    # Check correspondence between code and graph datasets
    print("Checking correspondence between code and graph datasets...")
    #
    assert len(dataset) == len(code_data) , f"Code dataset and graph dataset sizes do not match: {len(dataset)} != {len(code_data)}"

    
    # Initialize model
    print(f"Initializing model...")
    model = CodeGraphEncoder(**config['model'])
    
    # Modify model parameter loading section
    if not args.use_random_weights:
        # Load model parameters
        print(f"Loading model parameters from {args.model_path}...")
        try:
            checkpoint = torch.load(args.model_path, map_location='cpu')
            # Handle different forms of model saving
            if isinstance(checkpoint, dict) and all(not k.startswith("module.") for k in checkpoint.keys()):
                model.load_state_dict(checkpoint)
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            print("Model parameters loaded successfully!")
        except Exception as e:
            print(f"Error loading model parameters: {e}")
            return
    else:
        print("Using randomly initialized weights")
        # Optional: Reinitialize weights
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        model.apply(init_weights)
    
    # Move model to specified device
    model = model.to(device)
    model.eval()
    
    # Prepare data loader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Remove code encoder loading section
    # print(f"Loading code encoder: {args.code_encoder}")
    # tokenizer = RobertaTokenizer.from_pretrained(args.code_encoder)
    # encoder = RobertaModel.from_pretrained(args.code_encoder, device_map=device)
    # encoder.eval()  # Ensure in evaluation mode
    
    code_column_name = args.column_name
    
    # Check if processed files exist
    processed_indices = set()
    if os.path.exists(args.output_path):
        print(f"Found existing output file, reading processed indices...")
        try:
            with open(args.output_path, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    processed_indices.add(int(row[0]))  # Add processed idx
            print(f"Found {len(processed_indices)} processed samples")
            
            # Backup existing file
            backup_path = args.output_path + '.backup'
            os.rename(args.output_path, backup_path)
            print(f"Original file backed up as: {backup_path}")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            processed_indices = set()
    
    # Modify CSV header, remove code_emb column
    # header is left for backward compatibility, actual columns are handled by pandas
    header = ['idx', 'graph_emb', 'code']

    # If there are backup files, copy processed content
    if processed_indices:
        import shutil
        shutil.copy2(backup_path, args.output_path)
        file_mode = 'a'  # Append mode
    else:
        file_mode = 'w'  # New mode

    # First create a copy of all original data
    result_data = code_data.copy()
    
    # If there is no idx column, create one
    if 'idx' not in result_data.columns:
        result_data['idx'] = range(len(result_data))
    
    # Initialize output column(s)
    if args.output_type == 'graph':
        result_data['graph_emb'] = [None] * len(result_data)
    elif args.output_type == 'node':
        result_data['node_embs'] = [None] * len(result_data)
    
    # Process graph and code simultaneously in one pass
    print("Starting to extract code graph embedding vectors...")
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Processing batches")):
            data = data.to(device)
            
            # Get batch indices
            if hasattr(data, 'idx'):
                batch_indices = data.idx.cpu().tolist()
            else:
                batch_indices = list(range(i * args.batch_size, 
                                         min((i + 1) * args.batch_size, len(dataset))))
            
            # Skip processed batches
            if all(idx in processed_indices for idx in batch_indices):
                continue
            
            # Only process unprocessed samples
            unprocessed_indices = [idx for idx in batch_indices if idx not in processed_indices]
            if not unprocessed_indices:
                continue
                
            # Get node and graph embeddings
            node_emb, graph_emb = model(data.x, data.edge_index, data.edge_attr, data.batch)
            node_emb = node_emb.cpu()
            graph_emb = graph_emb.cpu().numpy()
            
            # Prepare per-graph node embedding slices if needed
            if args.output_type == 'node':
                if hasattr(data, 'ptr'):
                    ptr = data.ptr.cpu().tolist()
                else:
                    counts = torch.bincount(data.batch.cpu())
                    ptr = [0]
                    for c in counts.tolist():
                        ptr.append(ptr[-1] + c)
            
            for j, idx in enumerate(batch_indices):
                if idx not in processed_indices:
                    if idx >= len(result_data):
                        print(f"Warning: idx={idx} exceeds data range")
                        continue
                    
                    if args.output_type == 'graph':
                        result_data.at[idx, 'graph_emb'] = graph_emb[j].tolist()
                    elif args.output_type == 'node':
                        # Slice node embeddings belonging to this graph
                        if j + 1 >= len(ptr):
                            print(f"Warning: ptr length {len(ptr)} is insufficient for graph {j} in batch")
                            continue
                        start, end = ptr[j], ptr[j + 1]
                        result_data.at[idx, 'node_embs'] = node_emb[start:end].numpy().tolist()
                    
                    processed_indices.add(idx)
            
            # Clean up memory
            del graph_emb
            torch.cuda.empty_cache()
            
            if (i + 1) % 10 == 0:
                print(f"Processed {len(processed_indices)} samples")

    # Save final merged dataset
    result_data.to_csv(args.output_path, index=False)
    print(f"All data processing completed, saved to: {args.output_path}")
    print(f"Final dataset contains {len(result_data)} rows, fields: {list(result_data.columns)}")
    
    del model
    del dataloader
    torch.cuda.empty_cache()
    
    metadata = {
        'config': config,
        'edge_type_map': edge_type_map,
        'csv_path': args.output_path 
    }
    metadata_path = args.output_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    print("Completed!")

if __name__ == "__main__":
    main()
