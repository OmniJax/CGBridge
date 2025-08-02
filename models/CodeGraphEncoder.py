# Add the project root directory to the Python path
import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from .gnns import load_gnn_model
from torch_geometric.nn import GlobalAttention
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import global_mean_pool  # Import average pooling function

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class CodeGraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gnn_type='gt', 
                 num_layers=2, dropout=0.1, num_heads=8, edge_type_map=None, tau=0.5,
                 p_drop_node=0.1, p_drop_edge=0.1):
        super(CodeGraphEncoder, self).__init__()
        # Main backbone of the graph neural network
        self.gnn = load_gnn_model[gnn_type](
            in_channels=in_channels, 
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads
        )
        
        # Edge type mapping dictionary
        self.edge_type_map = edge_type_map
        if edge_type_map is None:
            self.edge_type_map = {
                0 : 'AST',
                1 : 'CFG',
                2 : 'DFG' 
            }
        self.num_edge_type = len(self.edge_type_map)
        self.edge_type_map[self.num_edge_type] = 'NO EDGE'
        print(f"edge_type_map: {self.edge_type_map}")
        print(f"num_edge_type: {self.num_edge_type}")
            
        # Edge predictor head (predicts the type of edges)
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(out_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, self.num_edge_type+1)  # n+1 types of edges
        )
        
        # Projection head for graph-level representation (used for contrastive learning)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(out_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Temperature parameter for contrastive learning
        self.tau = tau
        
        # Graph augmentation parameters
        self.p_drop_node = p_drop_node
        self.p_drop_edge = p_drop_edge
        
        # Using PyG's GlobalAttention
        # gate_nn = Sequential(Linear(out_channels, out_channels), 
        #                    ReLU(), 
        #                    Linear(out_channels, 1))
        # self.pool = GlobalAttention(gate_nn=gate_nn)
        self.pool = global_mean_pool
        
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        node_emb, _ = self.gnn(x, edge_index, edge_attr)
        
        if batch is not None:
            # Using PyG's GlobalAttention
            graph_emb = self.pool(node_emb, batch)
            return node_emb, graph_emb
        
        return node_emb, None
    
    def predict_edge_types(self, node_emb, edge_index):
        """Predict the types of edges"""
        src, dst = edge_index
        edge_features = torch.cat([node_emb[src], node_emb[dst]], dim=1)
        return self.edge_predictor(edge_features)
    
    def get_contrastive_loss(self, z1, z2):
        """Calculate contrastive loss (InfoNCE loss)"""
        # Project to contrastive learning space
        h1 = self.projector(z1)
        h2 = self.projector(z2)
        
        # Normalization - avoid in-place operations here
        h1 = F.normalize(h1, p=2, dim=1)
        h2 = F.normalize(h2, p=2, dim=1)
        
        # Calculate similarity matrix - use clone to avoid in-place modification
        sim_matrix = torch.matmul(h1, h2.T) / self.tau
        
        # Calculate InfoNCE loss - ensure labels are newly created
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss_a = F.cross_entropy(sim_matrix, labels)
        loss_b = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_a + loss_b) / 2.0
    
    def augment_graph(self, x, edge_index, edge_attr=None, p_drop_node=None, p_drop_edge=None):
        """Create a random augmented view of the graph"""
        # Use default values or passed values
        p_drop_node = p_drop_node if p_drop_node is not None else self.p_drop_node
        p_drop_edge = p_drop_edge if p_drop_edge is not None else self.p_drop_edge
        
        # Copy inputs
        x_aug = x.clone()
        edge_index_aug = edge_index.clone()
        
        # Handle edge attributes (if any)
        if edge_attr is not None:
            edge_attr_aug = edge_attr.clone()
        else:
            edge_attr_aug = None
        
        # Randomly drop some node features
        mask = torch.bernoulli(torch.ones_like(x_aug) * (1 - p_drop_node)).bool()
        x_aug = x_aug * mask
        
        # Randomly drop some edges
        if p_drop_edge > 0 and edge_index.size(1) > 0:
            num_edges = edge_index_aug.size(1)
            perm = torch.randperm(num_edges, device=edge_index.device)
            num_keep = int((1 - p_drop_edge) * num_edges)
            
            # Indices of edges to keep
            keep_indices = perm[:num_keep]
            
            # Update edge indices
            edge_index_aug = edge_index_aug[:, keep_indices]
            
            # Update edge attributes (if any)
            if edge_attr_aug is not None:
                edge_attr_aug = edge_attr_aug[keep_indices]
        
        return x_aug, edge_index_aug, edge_attr_aug