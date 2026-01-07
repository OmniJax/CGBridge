"""
CGBridge Stage 2: Graph-Text Cross-Modal Alignment
Based on BLIP-2 QFormer concepts.
Performs GTC, GTM, and GTG tasks.
"""

import logging
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.Qformer import BertConfig, BertLMHeadModel 
from transformers import BertTokenizer

# --- Distributed Training Utilities ---
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
        
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

def all_gather_with_grad(tensor):
    """
    Performs all_gather operation on the provided tensors with gradient propagation.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
        
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
        
    # Alias for backward pass
    class GatherLayer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor):
            ctx.batch_size = input_tensor.shape[0]
            gathered_tensor = [torch.zeros_like(input_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_tensor, input_tensor)
            gathered_tensor = torch.cat(gathered_tensor, 0)
            return gathered_tensor

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, async_op=False)
            # Return gradient for the local rank
            start_idx = dist.get_rank() * ctx.batch_size
            end_idx = start_idx + ctx.batch_size
            return grad_input[start_idx:end_idx]

    return GatherLayer.apply(tensor)


# --- Dataset ---
class CodeGraphTextDataset(Dataset):
    """Code Graph - Text Dataset, loads data from CSV."""
    
    def __init__(self, csv_path, max_length=512):
        """
        Args:
            csv_path: Path to CSV file with columns: idx, graph_emb or node_embs, code, code_emb
            max_length: Maximum sequence length for code text.
        """
        try:
            self.df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"Error loading CSV {csv_path}: {e}")
            raise
        self.max_length = max_length
        # Check whether node_embs is available (list of per-node embeddings)
        self.has_node_embs = 'node_embs' in self.df.columns

        # Determine code column name
        possible_code_columns = ('code', 'source_code', 'input_code')
        self.code_column = next((c for c in possible_code_columns if c in self.df.columns), None)
        if self.code_column is None:
            raise ValueError(
                f"Cannot find code column in CSV. Expected one of: {possible_code_columns}. "
                f"Available columns: {list(self.df.columns)}"
            )
        logging.info(f"Loaded {len(self.df)} samples from {csv_path}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns: (idx, graph_emb, code, code_emb)
        """
        item = self.df.iloc[idx]
        
        try:
            if self.has_node_embs:
                # node_embs is expected to be a JSON stringified list of lists
                node_embs = json.loads(item['node_embs'])
                graph_emb = torch.tensor(node_embs, dtype=torch.float)
            else:
                graph_emb_str = item['graph_emb'].strip('[]').split(',')
                graph_emb = torch.tensor([float(x) for x in graph_emb_str], dtype=torch.float)
            
            code = str(item[self.code_column]) # Ensure code is string
            
            # code_emb_str = item['code_emb'].strip('[]').split(',')
            # code_emb = torch.tensor([float(x) for x in code_emb_str], dtype=torch.float)
        except Exception as e:
             logging.error(f"Error processing item at index {idx} (CSV index: {item.get('idx', 'N/A')}): {e}")
             # Return dummy data or raise an error, depending on desired behavior
             # Returning dummy data might hide issues. Raising is safer.
             raise ValueError(f"Error processing data for item {idx}") from e

        # return item['idx'], graph_emb, code, code_emb
        return item['idx'], graph_emb, code, ' '

class CGBridgeStage2(nn.Module):
    """
    Stage 2 Pre-training for Code Graph and Text Alignment using BERT Q-Former.
    Includes GTC, GTM, and GTG losses with ablation study support.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary containing model parameters:
                - bert_model_dir: Path to the pre-trained BERT model directory.
                - num_query_token: Number of query tokens for Q-Former.
                - graph_width: Dimension of the input graph embeddings.
                - cross_attention_freq: Frequency of cross-attention layers.
                - embed_dim: Dimension for the shared projection space.
                - max_txt_len: Maximum length for text tokenization.
                - device: Device to run the model on ('cuda' or 'cpu').
        """
        super().__init__()
        
        model_config = config.get('model', config) # Allow passing model sub-config or full config
        
        self.device = torch.device(model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.max_txt_len = model_config['max_txt_len']
        bert_model_dir = model_config['bert_model_dir']
        num_query_token = model_config['num_query_token']
        graph_width = model_config['graph_width']
        cross_attention_freq = model_config['cross_attention_freq']
        embed_dim = model_config['embed_dim']
        self.num_query_token = num_query_token
        
        # Initialize Tokenizer
        self.tokenizer = self.init_tokenizer(bert_model_dir)
        
        # Initialize Q-Former and Query Tokens
        self.Qformer, self.query_tokens = self.init_Qformer(
            bert_model_dir, num_query_token, graph_width, cross_attention_freq
        )
        # Resize token embeddings if tokenizer vocab size changed (e.g., added tokens)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize query FFN layers by copying weights from standard FFN layers
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        # Projection Heads for Contrastive Learning
        self.graph_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        
        # Image-Text Matching (GTM) Head
        self.gtm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        
        # Temperature parameter for contrastive loss
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        
        # Add training objective switch configuration
        trainer_config = config.get('trainer', {})
        objectives_config = trainer_config.get('objectives', {})
        
        # GTC is always enabled, GTM and GTG can be controlled via configuration
        self.enable_gtc = True  # GTC is always trained
        self.enable_gtm = objectives_config.get('enable_gtm', True)
        self.enable_gtg = objectives_config.get('enable_gtg', True)
        
        # Move model components to the target device
        self.to(self.device)

    def init_tokenizer(self, tokenizer_dir):
        """Initializes the BERT tokenizer."""
        try:
            tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
            tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            logging.info(f"Tokenizer initialized from {tokenizer_dir}")
            # Log special token IDs
            logging.info(f"Tokenizer BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}, PAD: {tokenizer.pad_token_id}, CLS: {tokenizer.cls_token_id}, SEP: {tokenizer.sep_token_id}")
            return tokenizer
        except Exception as e:
            logging.error(f"Error initializing tokenizer from {tokenizer_dir}: {e}")
            raise

    def init_Qformer(self, bert_model_dir, num_query_token, graph_width, cross_attention_freq):
        """Initializes the Q-Former (BertLMHeadModel) and query tokens."""
        encoder_config = BertConfig.from_pretrained(bert_model_dir)
        encoder_config.encoder_width = graph_width
        # Insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(encoder_config)
        checkpoint = torch.load(os.path.join(bert_model_dir, "model.pth"), map_location=lambda storage, loc: storage)

        Qformer.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logging.info(f"Q-Former (BertLMHeadModel) initialized from {bert_model_dir} with custom config.")
        # Initialize query_tokens
        query_tokens = nn.Parameter(
            # 1, 32, 768
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def forward(self, samples):
        """
        Forward pass for Stage 2 pre-training with ablation study support.
        Args:
            samples: Tuple (idx, graph_emb, code, code_emb)
                - idx: Sample indices (not used in computation)
                - graph_emb: Batch of graph embeddings [batch_size, graph_emb_dim]
                - code: List of code strings [batch_size]
                - code_emb: Batch of code embeddings (not directly used by QFormer, maybe for other losses if added)
        Returns:
            Dictionary containing total loss and individual loss components (gtc, gtm, gtg).
        """
        _, graph_emb, code_text, _ = samples
        
        # Prepare graph embeddings (input Zv)
        # Support either a single graph-level embedding [bs, dim] or a sequence of node embeddings [bs, seq_len, dim]
        if isinstance(graph_emb, tuple):
            graph_embeds, graph_atts = graph_emb  # Expect (embeds, attention_mask)
        else:
            graph_embeds = graph_emb
            graph_atts = None
        
        if graph_embeds.dim() == 2:
            # Only one graph token per sample
            graph_embeds = graph_embeds.unsqueeze(1)
            graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long)
        elif graph_embeds.dim() == 3:
            # Already a sequence, create a full attention mask if not provided
            if graph_atts is None:
                graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long)
        else:
            raise ValueError(f"Unsupported graph_embeds shape: {graph_embeds.shape}")
        
        graph_embeds = graph_embeds.to(self.device)
        graph_atts = graph_atts.to(self.device)
        
        # Expand query tokens for the batch
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        
        # --- Q-Former Pass 1: Extract Graph Features using Queries ---
        # Input: Query Tokens (as query_embeds), Graph Embeddings (as encoder_hidden_states)
        # Output: Query outputs conditioned on graph embeddings (Hv = f(Q, Zv))
        # is_decoder=False here because query tokens attend to graph freely
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,          # Q
            encoder_hidden_states=graph_embeds, # Zv
            encoder_attention_mask=graph_atts,
            use_cache=True, # Important for GTG stage later
            return_dict=True,
            is_decoder=False # Queries attend fully to graph
        )
        # Shape: [batch_size, num_query_token, hidden_size]
        graph_query_features = query_output.last_hidden_state 

        # Project graph features for contrastive loss
        # Hv
        graph_feats_proj = F.normalize(self.graph_proj(graph_query_features), dim=-1)

        # --- Prepare Text Input ---
        text_tokens = self.tokenizer(
            code_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        
        # --- Q-Former Pass 2: Extract Text Features ---
        # Input: Text Tokens (as input_ids)
        # Output: Text embeddings (Tv = f(tv))
        # Standard BERT forward pass for text
        text_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
            is_decoder=False # Standard BERT encoder
        )
        # Use CLS token representation for text feature
        # Shape: [batch_size, hidden_size]
        text_cls_feature = text_output.last_hidden_state[:, 0, :] 
        
        # Project text features for contrastive loss
        text_feat_proj = F.normalize(self.text_proj(text_cls_feature), dim=-1)
        
        # --- CodeGraph-Text Contrastive Loss (GTC) - Always computed ---
        # Gather features across GPUs if distributed
        graph_feats_all = concat_all_gather(graph_feats_proj) # [bs * n_gpu, num_query, embed_dim]
        text_feat_all = concat_all_gather(text_feat_proj)     # [bs * n_gpu, embed_dim]
        
        # Calculate sim_q2t: similarity between graph queries and all texts
        # Matmul: [bs, num_query, embed_dim] x [bs * n_gpu, embed_dim].T -> [bs, num_query, bs * n_gpu]
        sim_q2t = torch.matmul(
            graph_feats_proj.unsqueeze(1), # [bs, 1, num_query, embed_dim]
            text_feat_all.unsqueeze(-1) # [bs * n_gpu, embed_dim, 1]
        ).squeeze() # [batch_size, batch_size*num_gpu, num_query_tokens]
        
        # Aggregate similarity across queries (max pooling) -> [bs, bs * n_gpu]
        sim_g2t, _ = sim_q2t.max(-1) 
        sim_g2t = sim_g2t / self.temp

        # Calculate sim_t2q: similarity between text CLS and all graph queries
        # Matmul: [bs, 1, embed_dim] x [bs * n_gpu, num_query, embed_dim].permute(0, 2, 1) -> [bs, 1, bs * n_gpu, num_query]
        # Squeeze to get [bs, bs * n_gpu, num_query]
        sim_t2q = torch.matmul(
            text_feat_proj.unsqueeze(1).unsqueeze(1),                   # [bs, 1, 1, embed_dim]
            graph_feats_all.permute(0, 2, 1) # [bs * n_gpu, embed_dim, num_query]
        ).squeeze() # Result shape: [bs, bs * n_gpu, num_query]

        # Aggregate similarity across queries (max pooling) -> [bs, bs * n_gpu]
        sim_t2g, _ = sim_t2q.max(-1) 
        sim_t2g = sim_t2g / self.temp
        
        # Create targets for contrastive loss (positive pairs are on the diagonal)
        bs = graph_embeds.size(0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # targets = torch.arange(bs, dtype=torch.long, device=self.device) + rank * bs
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)
        loss_gtc = (
            F.cross_entropy(sim_g2t, targets, label_smoothing=0.1) +
            F.cross_entropy(sim_t2g, targets, label_smoothing=0.1)
        ) / 2

        # --- CodeGraph-Text Matching Loss (GTM) - Optional ---
        if self.enable_gtm:
            # Complete GTM calculation logic
            # Gather necessary tensors across GPUs for negative sampling
            text_input_ids_world = concat_all_gather(text_tokens.input_ids)
            text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
            graph_embeds_world = all_gather_with_grad(graph_embeds) 
            
            # Calculate weights for hard negative mining
            with torch.no_grad():
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
                weights_t2g[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_g2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

            # Select hard negative graph for each text
            graph_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(graph_embeds_world[neg_idx])
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)

            # Select hard negative text for each graph
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(text_input_ids_world[neg_idx])
                text_atts_neg.append(text_attention_mask_world[neg_idx])
            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)
            
            # Combine positive and negative samples
            graph_embeds_all = torch.cat([graph_embeds, graph_embeds_neg, graph_embeds], dim=0)
            graph_atts_all = torch.ones(graph_embeds_all.size()[:-1], dtype=torch.long).to(self.device)
            text_ids_all = torch.cat([text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0)
            text_atts_all = torch.cat([text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg], dim=0)

            # Prepare inputs for Q-Former GTM pass
            query_tokens_gtm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_atts_gtm = torch.ones(query_tokens_gtm.size()[:-1], dtype=torch.long).to(self.device)
            attention_mask_all = torch.cat([query_atts_gtm, text_atts_all], dim=1)

            # --- Q-Former Pass 3: Multimodal Fusion for GTM ---
            output_gtm = self.Qformer.bert(
                input_ids=text_ids_all,
                query_embeds=query_tokens_gtm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                return_dict=True,
                is_decoder=False
            )
            
            # Extract features and calculate logits (always computed)
            gl_embeddings = output_gtm.last_hidden_state[:, :self.num_query_token, :] 
            gtm_logits = self.gtm_head(gl_embeddings).mean(dim=1)
            
            # Create labels
            gtm_labels = torch.cat(
                [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], 
                dim=0
            ).to(self.device)
            
            # Calculate loss, but whether to use it depends on configuration
            loss_gtm = F.cross_entropy(gtm_logits, gtm_labels)
        else:
            # If GTM is disabled, set loss to 0 (do not perform computation)
            loss_gtm = torch.tensor(0.0, device=self.device, requires_grad=True)

        # --- Language Modeling Loss (GTG) - Optional ---
        if self.enable_gtg:
            # Complete GTG calculation logic
            decoder_input_ids = text_tokens.input_ids.clone()
            start_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id
            decoder_input_ids[:, 0] = start_token_id

            labels = decoder_input_ids.masked_fill(
                decoder_input_ids == self.tokenizer.pad_token_id, -100
            )

            gtg_query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            gtg_attention_mask = torch.cat([gtg_query_atts, text_tokens.attention_mask], dim=1)

            # --- Q-Former Pass 4: Language Modeling (always computed) ---
            gtg_output = self.Qformer(
                input_ids=decoder_input_ids,
                attention_mask=gtg_attention_mask,
                past_key_values=query_output.past_key_values,
                return_dict=True,
                labels=labels,
                is_decoder=True
            )
            
            # Whether to use GTG loss depends on configuration
            loss_gtg = gtg_output.loss
        else:
            # If GTG is disabled, set loss to 0 (do not perform computation)
            loss_gtg = torch.tensor(0.0, device=self.device, requires_grad=True)

        # --- Total Loss ---
        total_loss = loss_gtc
        if self.enable_gtm:
            total_loss = total_loss + loss_gtm
        if self.enable_gtg:
            total_loss = total_loss + loss_gtg
        
        return {
            "loss": total_loss,
            "loss_gtc": loss_gtc,
            "loss_gtm": loss_gtm,
            "loss_gtg": loss_gtg,
        }

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=512, # Max length of generated sequence
        min_length=10,  # Min length of generated sequence
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0, # Use > 1.0 to encourage longer sequences
        num_captions=1, # Number of captions per graph
        temperature=1.0, # For sampling
    ):
        """
        Generate code text conditioned on graph embeddings.
        Args:
            samples: Tuple (idx, graph_emb, code, code_emb) - only graph_emb is used.
            use_nucleus_sampling: Boolean, true for nucleus sampling, false for beam search.
            num_beams: Number of beams for beam search.
            max_length: Maximum length of the generated text.
            min_length: Minimum length of the generated text.
            top_p: Top-p probability for nucleus sampling.
            repetition_penalty: Penalty for repeating tokens.
            length_penalty: Penalty factor for sequence length in beam search.
            num_captions: Number of captions to generate per input graph.
            temperature: Temperature for sampling.
        Returns:
            List of generated code strings.
        """
        _, graph_emb, _, _ = samples
        
        # Support graph-level or node-level embeddings at generation time as well
        if isinstance(graph_emb, tuple):
            graph_embeds, graph_atts = graph_emb
        else:
            graph_embeds = graph_emb
            graph_atts = None
        
        if graph_embeds.dim() == 2:
            graph_embeds = graph_embeds.unsqueeze(1)
            graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long)
        elif graph_embeds.dim() == 3:
            if graph_atts is None:
                graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long)
        else:
            raise ValueError(f"Unsupported graph_embeds shape: {graph_embeds.shape}")

        graph_embeds = graph_embeds.to(self.device)
        graph_atts = graph_atts.to(self.device)
        bs = graph_embeds.shape[0]

        # --- Q-Former Pass 1: Get Graph Context ---
        # Same as in forward pass to get graph features encoded into query tokens
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_atts,
            use_cache=True, # We need past_key_values
            return_dict=True,
            is_decoder=False # Queries attend fully to graph
        )
        graph_past_key_values = query_output.past_key_values

        # --- Prepare for Generation ---
        # Start token: typically [CLS] for BERT LM, or a dedicated [BOS] if defined
        start_token_id = self.tokenizer.cls_token_id 
        input_ids = torch.full((bs * num_captions, 1), start_token_id, dtype=torch.long).to(self.device)
        
        # Attention mask starts with just the start token
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(self.device)

        # Expand graph context (past_key_values) for beam search/multiple captions
        graph_past_key_values = self._expand_past(graph_past_key_values, bs, num_beams * num_captions)
        
        # Generation arguments
        gen_kwargs = {
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "do_sample": use_nucleus_sampling,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "eos_token_id": self.tokenizer.sep_token_id, # Use [SEP] as EOS for BERT
            "pad_token_id": self.tokenizer.pad_token_id,
            # Pass the graph context
            "past_key_values": graph_past_key_values, 
            # Provide attention mask matching the graph context shape
            "attention_mask": torch.ones(bs * num_beams * num_captions, graph_past_key_values[0][0].shape[-2], device=self.device, dtype=torch.long)
        }
        
        # --- Generate ---
        # Note: Qformer.generate needs to handle past_key_values correctly.
        # The BertLMHeadModel's generate method should work if `is_decoder=True` is handled internally or passed.
        # We pass the graph context via past_key_values. The model's generate loop will append the text's self-attention KV cache.
        outputs = self.Qformer.generate(
            input_ids=input_ids, # Start token(s)
            attention_mask=attention_mask, # Initial mask for start token(s)
            # The graph context is passed via past_key_values in gen_kwargs
            is_decoder=True, # Ensure causal masking is used during generation
            **gen_kwargs
        )

        # Decode output tokens
        # Remove the input_ids part (start token) if included in output
        if outputs.shape[1] > 1 and torch.all(outputs[:, 0] == start_token_id):
            output_sequences = outputs[:, 1:]
        else:
            output_sequences = outputs
            
        captions = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        return captions
        
    def _expand_past(self, past_key_values, batch_size, num_expand):
        """Expand past_key_values for beam search or multiple captions."""
        if past_key_values is None or num_expand == 1:
            return past_key_values
            
        # Correct expansion based on beam search/num_captions logic
        # Each item in past_key_values tuple corresponds to a layer
        # Each layer tuple contains (key, value) tensors: [batch_size, num_heads, seq_len, head_dim]
        
        # Calculate expansion factor per item in batch
        expand_factor = num_expand // batch_size
        if num_expand % batch_size != 0:
             raise ValueError("num_expand must be a multiple of batch_size")

        new_past = []
        for layer_past in past_key_values:
            new_layer_past = []
            for state in layer_past: # key or value
                # state shape: [batch_size, num_heads, seq_len, head_dim]
                # Expand shape: [batch_size * expand_factor, num_heads, seq_len, head_dim]
                new_state = state.repeat_interleave(expand_factor, dim=0)
                new_layer_past.append(new_state)
            new_past.append(tuple(new_layer_past))
            
        return tuple(new_past)

    def load_from_pretrained(self, url_or_filename):
        """Loads checkpoint, compatible with common saving formats."""
        if not url_or_filename:
            logging.warning("No checkpoint path provided for loading.")
            return None
            
        if not os.path.exists(url_or_filename):
             logging.error(f"Checkpoint file not found: {url_or_filename}")
             return None
             
        try:
            checkpoint = torch.load(url_or_filename, map_location=lambda storage, loc: storage.to(self.device))
            
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                 state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint # Assume the checkpoint is the state dict itself
            
            # Handle potential prefix issues (e.g., 'module.' if saved with DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load the state dict
            msg = self.load_state_dict(state_dict, strict=False)
            logging.info(f"Loaded checkpoint from {url_or_filename}. Load message: {msg}")
            return msg
        except Exception as e:
            logging.error(f"Error loading checkpoint from {url_or_filename}: {e}")
            raise 
