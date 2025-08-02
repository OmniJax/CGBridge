"""
 Stage 1: CGE Pre-training
"""
import sys
import os
import gc
import shutil
import yaml
import time
import argparse
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm 
from torch_geometric.utils import batched_negative_sampling

from accelerate import Accelerator

from scripts.utils import outputs_dir, load_config
from models.CodeGraphEncoder import CodeGraphEncoder

class CodeGNNTrainer:
    def __init__(self, model, optimizer, scheduler, train_loader, valid_loader, accelerator,
                 save_dir='checkpoints', lambda_contrast=0.5, lambda_edge_type=0.5,
                 p_drop_node=0.1, p_drop_edge=0.05, neg_samples_ratio=0.5):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.accelerator = accelerator
        
        self.save_dir = save_dir
        self.lambda_contrast = lambda_contrast
        self.lambda_edge_type = lambda_edge_type
        
        self.p_drop_node = p_drop_node
        self.p_drop_edge = p_drop_edge
        self.neg_samples_ratio = neg_samples_ratio
        
        if self.accelerator.is_main_process:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_filename = f'training_{timestamp}.log'
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(message)s',
                handlers=[
                    logging.FileHandler(os.path.join(save_dir, log_filename)),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())

    def _forward_pass(self, data):
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        x_aug1, edge_index_aug1, edge_attr_aug1 = unwrapped_model.augment_graph(
            data.x, data.edge_index, data.edge_attr,
            p_drop_node=self.p_drop_node, p_drop_edge=self.p_drop_edge
        )
        x_aug2, edge_index_aug2, edge_attr_aug2 = unwrapped_model.augment_graph(
            data.x, data.edge_index, data.edge_attr,
            p_drop_node=self.p_drop_node, p_drop_edge=self.p_drop_edge
        )
        
        _, z1 = self.model(x_aug1, edge_index_aug1, edge_attr_aug1, data.batch)
        _, z2 = self.model(x_aug2, edge_index_aug2, edge_attr_aug2, data.batch)
        
        cl_loss = unwrapped_model.get_contrastive_loss(z1, z2)
        
        del x_aug1, edge_index_aug1, edge_attr_aug1, x_aug2, edge_index_aug2, edge_attr_aug2, z1, z2
        
        node_emb, _ = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        edge_index = data.edge_index
        num_edges = 0
        
        if edge_index is not None and edge_index.numel() > 0:
            if edge_index.dim() == 2 and edge_index.size(0) == 2:
                num_edges = edge_index.size(1)
            else:
                print(f"Warning: edge_index dimension is abnormal: {edge_index.shape}")
        
        if num_edges == 0:
            edge_loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
        else:
            edge_pred = unwrapped_model.predict_edge_types(node_emb, data.edge_index)
            
            if hasattr(data, 'edge_type') and data.edge_type is not None:
                edge_type = data.edge_type
            else:
                edge_type = torch.zeros(num_edges, dtype=torch.long, device=self.accelerator.device)
            
            if edge_type.numel() == 0:
                edge_loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
            else:
                edge_loss = F.cross_entropy(edge_pred, edge_type)
            
            del edge_pred
            
            num_neg_samples = max(1, int(num_edges * self.neg_samples_ratio))  
            try:
                neg_edge_index = batched_negative_sampling(
                    edge_index=data.edge_index,
                    batch=data.batch,
                    num_neg_samples=num_neg_samples,
                    method="sparse"
                )
                
                neg_edge_pred = unwrapped_model.predict_edge_types(node_emb, neg_edge_index)
                
                neg_edge_type = torch.full((neg_edge_index.size(1),), unwrapped_model.num_edge_type,
                                          dtype=torch.long, device=self.accelerator.device)
                
                neg_edge_loss = F.cross_entropy(neg_edge_pred, neg_edge_type)
                
                del neg_edge_index, neg_edge_pred, neg_edge_type
                
                edge_loss = (edge_loss + neg_edge_loss) / 2
            except Exception as e:
                print(f"Negative sampling failed, using only positive sample loss: {e}")
        
        del node_emb
        
        loss = self.lambda_contrast * cl_loss + self.lambda_edge_type * edge_loss
        
        return loss, cl_loss, edge_loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_cl_loss = 0
        total_edge_loss = 0
        
        for data in tqdm(self.train_loader, desc="Training", disable=not self.accelerator.is_main_process):
            self.optimizer.zero_grad()
            
            loss, cl_loss, edge_loss = self._forward_pass(data)
            
            self.accelerator.backward(loss)
            
            self.optimizer.step()
            
            total_loss += self.accelerator.gather(loss).sum().item()
            total_cl_loss += self.accelerator.gather(cl_loss).sum().item()
            total_edge_loss += self.accelerator.gather(edge_loss).sum().item()

        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_cl_loss = total_cl_loss / len(self.train_loader.dataset)
        avg_edge_loss = total_edge_loss / len(self.train_loader.dataset)

        return avg_loss, {'cl_loss': avg_cl_loss, 'edge_loss': avg_edge_loss}

    def eval_epoch(self):
        self.model.eval()
        total_loss = 0
        total_cl_loss = 0
        total_edge_loss = 0
        
        with torch.no_grad():
            for data in tqdm(self.valid_loader, desc="Evaluating", disable=not self.accelerator.is_main_process):
                loss, cl_loss, edge_loss = self._forward_pass(data)
                
                total_loss += self.accelerator.gather(loss).sum().item()
                total_cl_loss += self.accelerator.gather(cl_loss).sum().item()
                total_edge_loss += self.accelerator.gather(edge_loss).sum().item()
        
        avg_loss = total_loss / len(self.valid_loader.dataset)
        avg_cl_loss = total_cl_loss / len(self.valid_loader.dataset)
        avg_edge_loss = total_edge_loss / len(self.valid_loader.dataset)
        
        return avg_loss, {'cl_loss': avg_cl_loss, 'edge_loss': avg_edge_loss}

    def train(self, num_epochs=100, patience=10, monitor='loss', mode='min', save_path=None):
        early_stopping = EarlyStopping(
            patience=patience, mode=mode, logger=self.logger, save_dir=self.save_dir
        ) if self.accelerator.is_main_process else None
        
        for epoch in range(num_epochs):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.eval_epoch()
            val_loss_gathered = self.accelerator.gather(torch.tensor(val_loss, device=self.accelerator.device)).mean().item()
            self.scheduler.step(val_loss_gathered)
            
            if self.accelerator.is_main_process:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}:")
                self.logger.info(f"  Training loss: {train_loss:.6f}")
                for k, v in train_metrics.items():
                    self.logger.info(f"  Training {k}: {v:.6f}")
                self.logger.info(f"  Validation loss: {val_loss:.6f}")
                for k, v in val_metrics.items():
                    self.logger.info(f"  Validation {k}: {v:.6f}")
                
                if monitor == 'loss':
                    monitor_value = val_loss
                else:
                    monitor_value = val_metrics[monitor]
                
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                early_stopping(monitor_value, unwrapped_model)
                
                if early_stopping.early_stop:
                    self.logger.info(f"Training stopped early at epoch {epoch+1}")
                    early_stopping.load_best_model(unwrapped_model)
            self.accelerator.wait_for_everyone()
            if early_stopping and early_stopping.early_stop:
                 break
            
            gc.collect()
            torch.cuda.empty_cache()
        
        if self.accelerator.is_main_process and save_path is not None:
            self.accelerator.wait_for_everyone()
            final_save_path = save_path / "final_model.pt"
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            if early_stopping and early_stopping.best_model_state is not None:
                state_dict_to_save = early_stopping.best_model_state
            else:
                state_dict_to_save = unwrapped_model.state_dict()
            
            self.accelerator.save(state_dict_to_save, final_save_path)
            self.logger.info(f"Model saved to {final_save_path}")

        if self.accelerator.is_main_process:
            self.logger.info("Training completed")
    
    def extract_graph_representations(self, dataset, batch_size=32):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # Prepare the data loader for distributed evaluation
        loader = self.accelerator.prepare(loader)
        
        all_graph_embs = []
        
        with torch.no_grad():
            for data in tqdm(loader, desc="Extracting graph embeddings", disable=not self.accelerator.is_main_process):
                _, graph_emb = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                graph_emb_gathered = self.accelerator.gather_for_metrics(graph_emb)
                all_graph_embs.append(graph_emb_gathered.cpu())
                
        return torch.cat(all_graph_embs, dim=0)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min', verbose=True, logger=None, save_dir=None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.logger = logger
        self.save_dir = save_dir

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            return
            
        improved = False
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                improved = True
        else: # mode == 'max'
            if score > self.best_score + self.min_delta:
                improved = True

        if improved:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
            if self.save_dir:
                torch.save(model.state_dict(), os.path.join(self.save_dir, "cur_best_model.pt"))
                if self.logger:
                    self.logger.info(f"The current best model has been saved. New score: {self.best_score:.6f}")
        else:
            self.counter += 1
                
        if self.counter >= self.patience:
            if self.verbose and self.logger:
                self.logger.info(f"Early stopping triggered! Best score: {self.best_score:.6f}")
            self.early_stop = True
            
    def save_checkpoint(self, model):
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

def load_datasets(train_path, val_path):
    print(f"Loading training set: {train_path}")
    train_data = torch.load(train_path, weights_only=False)
    
    print(f"Loading validation set: {val_path}")
    val_data = torch.load(val_path, weights_only=False)
    
    # Filter out graphs without edges
    def filter_graphs_with_edges(graphs):
        filtered = []
        skipped = 0
        for i, graph in enumerate(graphs):
            try:
                # Safety check for edge_index dimensions and shape
                if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                    edge_index = graph.edge_index
                    
                    # Check edge_index dimensions
                    if edge_index.dim() == 2 and edge_index.size(0) == 2 and edge_index.size(1) > 0:
                        # Ensure edge_type also exists and is of correct data type
                        if hasattr(graph, 'edge_type') and graph.edge_type is not None:
                            # Ensure edge_type is of type torch.long
                            if graph.edge_type.dtype != torch.long:
                                graph.edge_type = graph.edge_type.long()
                        else:
                            # If no edge_type, create a default one
                            graph.edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
                        filtered.append(graph)
                    else:
                        skipped += 1
                        print(f"Skipping graph {i}: edge_index shape is abnormal {edge_index.shape if hasattr(edge_index, 'shape') else 'None'}")
                else:
                    skipped += 1
                    print(f"Skipping graph {i}: no edge_index attribute")
            except Exception as e:
                skipped += 1
                print(f"Skipping graph {i}: error during processing {e}")
        
        print(f"Skipped {skipped} problematic graphs, kept {len(filtered)} valid graphs")
        return filtered
    
    train_graphs = filter_graphs_with_edges(train_data['graphs'])
    val_graphs = filter_graphs_with_edges(val_data['graphs'])
    
    return train_graphs, val_graphs, train_data.get('edge_label_to_name')

if __name__ == "__main__":
    # Initialize Accelerator    
    accelerator = Accelerator()
    
    parser = argparse.ArgumentParser(description='Train the code graph encoder')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to resume training from checkpoint')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to configuration file')
    args = parser.parse_args()
    
    config_path = args.config
    config = load_config(config_path)

    # Only the main process prints and saves files
    if accelerator.is_main_process:
        print(f"config: {config}")
        print('--------------------------------')
    
    train_path = config['data']['train_path']
    valid_path = config['data']['valid_path']
    save_dir = outputs_dir() / "checkpoints" / config['output']['save_dir']
    
    # After loading the dataset, integrate the edge type mapping into the configuration
    train_dataset, val_dataset, edge_type_map = load_datasets(train_path, valid_path)
    if accelerator.is_main_process:
        print(f"Edge type mapping loaded from dataset: {edge_type_map}")

    # Update the edge type mapping in the main configuration
    config['model']['edge_type_map'] = edge_type_map

    # Now save the updated configuration file
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        try:
            # Save the updated configuration regardless of the source
            config_save_path = save_dir / "config.yaml"
            with open(config_save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"Updated configuration file saved to: {config_save_path}")
            
            # Create a timestamped backup
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_config_path = save_dir / f"config_{timestamp}.yaml"
            with open(backup_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"Configuration file backup saved to: {backup_config_path}")
            
            # If the original configuration file exists, create a backup of it as well
            if os.path.exists(config_path):
                original_backup_path = save_dir / f"original_config_{timestamp}.yaml"
                shutil.copy2(config_path, original_backup_path)
                print(f"Original configuration file backup saved to: {original_backup_path}")
                
        except Exception as e:
            print(f"Error saving configuration file: {e}")

    # Wait on all processes to ensure the main process has created the directory and configuration file
    accelerator.wait_for_everyone()

    # Create the model - now config['model'] contains the correct edge_type_map
    model = CodeGraphEncoder(**config['model'])

    if accelerator.is_main_process:
        print(f"Edge type mapping used by the model: {model.edge_type_map}")
        print(f"Number of edge types: {model.num_edge_type}")
    
    if args.resume:
        if accelerator.is_main_process:
            print(f"Loading model parameters from {args.resume}...")
        try:
            # Load state before passing the model to prepare
            checkpoint = torch.load(args.resume, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            if accelerator.is_main_process:
                print("Model parameters loaded successfully!")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Error loading model parameters: {e}")
                print("Continuing training with a randomly initialized model.")
    else:
        if accelerator.is_main_process:
            print("Starting new training")

    # Create data loaders, optimizer, and scheduler
    train_loader = DataLoader(train_dataset, batch_size=config['trainer']['batch_size'], shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_dataset, batch_size=config['trainer']['batch_size'], shuffle=False, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['trainer']['lr'], weight_decay=config['trainer']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Wrap all components with accelerator.prepare()
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
    
    trainer = CodeGNNTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        accelerator=accelerator,
        save_dir=save_dir,
        lambda_contrast=config['trainer']['lambda_contrast'],
        lambda_edge_type=config['trainer']['lambda_edge_type'],
        neg_samples_ratio=config['trainer']['neg_samples_ratio']
    )
    
    trainer.train(
        num_epochs=config['trainer']['num_epochs'],
        patience=config['trainer']['patience'],
        monitor=config['trainer']['monitor'],
        mode=config['trainer']['mode'],
        save_path=save_dir
    )
    
    
"""

accelerate launch --num_processes 8 --gpu_ids="0,1,2,3,4,5,6,7" CGE_Trainer.py --config  /path/to/configs/CGE_configs/gat-ACD-2.yaml --resume /path/to/outputs/checkpoints/gat-ACD-unixcoder-2/cur_best_model.pt

"""