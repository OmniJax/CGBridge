"""
Stage 3: Instruction-Based Task Adaptation.
Implementation of code summarization task based on Stage 2 CGBridge and Qwen2.5-Coder-1.5B-Instruct
"""
import os
import sys
import re
import yaml
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, BertTokenizer

import contextlib

@contextlib.contextmanager
def no_init_weights(_):
    """
    Custom no_init_weights context manager to replace the version in transformers.utils.
    Used to temporarily skip weight initialization during model initialization.
    """
    yield

from models.CGBridge_Stage2 import CGBridgeStage2, CodeGraphTextDataset 

logger = logging.getLogger(__name__)


class CGBridgeStage3(nn.Module):
    """
    Stage 3: Code Graph-LLM Alignment Model
    """
    
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        
        # Load necessary parameters from Stage2 configuration
        self.load_stage2_config()
        
        # Device
        # self.device = torch.device(config["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.language = self.config.get("language", "Python")
        self.task = self.config.get("task", "summarization")
        
        # Configuration parameters
        self.stage2_model_dir = self.config["model"].get("stage2_model_dir", "")
        self.bert_model_dir = self.config["model"].get("bert_model_dir", "")
        self.llm_dir = self.config["model"]["llm_name_or_path"]
        self.llm_type = self.config["model"].get("llm_type", "auto").lower()
        self.max_txt_len = self.config["model"].get("max_txt_len", 512)
        self.num_query_token = self.config["model"].get("num_query_token", 32)
        self.num_features = self.config["model"].get("num_features", 768)
        
        # Initialize BERT tokenizer
        self.tokenizer = self.init_tokenizer()
        
        # Initialize Qformer and query_tokens
        self.Qformer, self.query_tokens = self.init_Qformer(
            self.num_query_token, self.num_features
        )
        print(f"Query tokens requires grad: {self.query_tokens.requires_grad}")
        
        # If needed, adjust tokenizer size
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None  # Remove classification head
        
        # Initialize LLM and tokenizer
        self.llm_tokenizer, self.llm_model = self._init_llm()

        logger.info(f"LLM precision: {self.llm_model.dtype}")
        # Define when class is initialized
        self.graph_token_id = self.llm_tokenizer.convert_tokens_to_ids("<|graph|>")
        
        # Freeze LLM parameters (except Qformer)
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # Projection layer: Project Q-Former output to LLM embedding space
        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )
        
        # Ensure model and parameters are on the current process's device
        # self.to(self.device)

    def _normalize_graph_inputs(self, graph_embs, device):
        """
        Normalize graph embeddings to shape [bs, seq_len, dim] and build attention mask.
        Supports either a single graph vector per sample or a padded node sequence.
        """
        if isinstance(graph_embs, tuple):
            graph_embeds, graph_atts = graph_embs
        else:
            graph_embeds, graph_atts = graph_embs, None

        if graph_embeds.dim() == 2:
            graph_embeds = graph_embeds.unsqueeze(1)
            if graph_atts is None:
                graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long)
        elif graph_embeds.dim() == 3:
            if graph_atts is None:
                graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long)
        else:
            raise ValueError(f"Unsupported graph_embeds shape: {graph_embeds.shape}")

        graph_embeds = graph_embeds.to(device)
        graph_atts = graph_atts.to(device=device, dtype=torch.long)
        return graph_embeds, graph_atts

    def load_stage2_config(self):
        """Load necessary parameters from Stage2 model configuration file"""
        import os
        import yaml
        
        # Ensure stage2_model_dir exists
        self.stage2_model_dir = self.config.get('stage2_model_dir', self.config.get('model', {}).get('stage2_model_dir', ''))
        if not self.stage2_model_dir:
            print("Warning: stage2_model_dir is not provided, cannot load Stage2 configuration")
            return
        
        # If path ends with best_model, take its parent directory
        stage2_dir = self.stage2_model_dir
        if stage2_dir.endswith('/best_model'):
            stage2_dir = os.path.dirname(stage2_dir)
        
        # Save as member variable
        self.stage2_config_path = os.path.join(stage2_dir, 'config.yaml')
        
        if not os.path.exists(self.stage2_config_path):
            print(f"Warning: Stage2 configuration file does not exist: {self.stage2_config_path}")
            return
        
        # Load Stage2 configuration
        try:
            with open(self.stage2_config_path, 'r') as f:
                stage2_config = yaml.safe_load(f)
            
            # Store complete stage2 configuration
            self.stage2_config = stage2_config
            
            # Extract model part from Stage2 configuration
            stage2_model_config = stage2_config.get('model', {})
            
            # List of key parameters to inherit from Stage2
            key_params = [
                'bert_model_dir',  # BERT-related
                'num_query_token', 'graph_width', 'cross_attention_freq', 'embed_dim'  # Q-Former-related
            ]
            
            # Inherit parameters from Stage2 that are missing in current configuration
            for key in key_params:
                if key not in self.config and key in stage2_model_config:
                    self.config['model'][key] = stage2_model_config[key]
                    print(f"Loaded parameter from Stage2 configuration: {key}={stage2_model_config[key]}")
            
            print(f"Successfully loaded parameters from Stage2 configuration: {self.stage2_config_path}")
        except Exception as e:
            print(f"Error loading Stage2 configuration: {e}")

    def init_tokenizer(self):
        """Initialize tokenizer, consistent with reference code"""
        tokenizer = BertTokenizer.from_pretrained(self.bert_model_dir)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    def init_Qformer(self, num_query_token, vision_width, cross_attention_freq=2):
        from models.CGBridge_Stage2 import CGBridgeStage2
        
        temp_stage2 = CGBridgeStage2(self.config)
        
        # get Qformer and query_tokens 
        Qformer = temp_stage2.Qformer
        query_tokens = temp_stage2.query_tokens
        
        # Load weights (if needed)
        if self.stage2_model_dir:
            self._load_qformer_weights(Qformer)
        
        del temp_stage2
        import gc
        gc.collect()  
        torch.cuda.empty_cache()  
        return Qformer, query_tokens
    
    def _load_qformer_weights(self, Qformer):
        """Load Qformer weights from Stage2 model directory"""
        logger.info(f"Loading Qformer weights from Stage2: {self.stage2_model_dir}")
        
        if not os.path.exists(self.stage2_model_dir):
            logger.warning(f"The specified Stage2 model path does not exist: {self.stage2_model_dir}")
            return
        
        checkpoint_path = None
        for possible_name in ["pytorch_model.bin", "model.safetensors", "model.bin"]:
            path = os.path.join(self.stage2_model_dir, possible_name)
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if not checkpoint_path:
            logger.warning(f"Model checkpoint file not found in the directory: {self.stage2_model_dir}")
            return
        
        try:
            if checkpoint_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                
                # Handle possible different formats
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
            
            # Filter out Qformer-related weights
            qformer_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("Qformer."):
                    qformer_state_dict[k.replace("Qformer.", "")] = v
                elif k.startswith("query_tokens"):
                    # Directly load query_tokens
                    pass
            
            # Load into Qformer
            missing, unexpected = Qformer.load_state_dict(qformer_state_dict, strict=False)
            
            if missing:
                logger.warning(f"Missing parameters when loading Qformer: {missing}")
            if unexpected:
                logger.warning(f"Unexpected parameters when loading Qformer: {unexpected}")
                
            logger.info(f"Successfully loaded Qformer weights: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error loading Qformer weights: {str(e)}", exc_info=True)
        
    def _init_llm(self):
        """Initialize language model, supporting various LLMs"""
        logger.info(f"Loading LLM model: {self.llm_dir}, type: {self.llm_type}")
        
        # Quantization configuration
        quantization_config = None
        if self.config["model"].get("quantize_llm", False):
            quantization_bit = self.config["model"].get("quantization_bit", 4)
            logger.info(f"Loading model with {quantization_bit}-bit quantization")
            
            if quantization_bit == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            elif quantization_bit == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
        
        # Use AutoTokenizer and AutoModelForCausalLM uniformly
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.llm_dir, 
            trust_remote_code=True,
            use_fast=False
        )
        tokenizer.padding_side = "left"
        
        # Ensure tokenizer has pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Add special token for graph representation
        if "<|graph|>" not in tokenizer.get_vocab():
            logger.info(f"Adding <|graph|> special token to tokenizer")
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|graph|>"]})
        
        # Get LLM data type
        llm_dtype = self.config["model"].get("llm_dtype", "bfloat16")
        if llm_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif llm_dtype == "float16":
            torch_dtype = torch.float16
        elif llm_dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported LLM data type: {llm_dtype}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )
        # After model.resize_token_embeddings(len(tokenizer))
        if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"Updating model.config.pad_token_id to: {tokenizer.pad_token_id}")
        
        # Adjust model to fit new tokens
        model.resize_token_embeddings(len(tokenizer))
        
        return tokenizer, model
    
    def prepare_lm_input(self, gtokens, text_input, answer=None):
        """
        Prepare LLM input, integrate instruction template processing, <|graph|> placed inside user message
        Args:
            gtokens: Q-Former output features [bsz, n_query, hidden_dim]
            text_input: Original code text list
            answer: Code summary text list (for training)
        """
        bsz, nvtoken, _ = gtokens.size()
        tokenizer = self.llm_tokenizer
        device = gtokens.device
        
        # Get task configuration
        language = self.language
        task = self.task
        prompts_config = self.config.get("prompts", {})
        task_prompts = prompts_config.get(task, {})
        
        system_prompt_template = ""
        user_prompt_template = ""
        if task_prompts:
            system_prompt_template = task_prompts.get("system_prompt", "")
            user_prompt_template = task_prompts.get("user_prompt", "")
        
        llm_type = self.llm_type
        graph_token_placeholder_str = "<|graph|>" # The string placeholder
        graph_placeholder = graph_token_placeholder_str * nvtoken

        full_input_texts = []
        prompt_token_lengths = []

        for i in range(bsz):
            code = text_input[i] # text_input is expected to be a list of strings

            current_system_prompt = system_prompt_template.format(language=language)

            user_content_with_graph = ""
            if user_prompt_template: # Ensure user_prompt_template is not empty
                user_content = user_prompt_template.format(language=language, code=code)
                user_content_with_graph = f"{graph_placeholder}\n{user_content}"
            else: # Fallback if no user_prompt_template
                user_content_with_graph = f"{graph_placeholder}\n{code}"
            
            prompt_text_formatted = ""
            if llm_type == 'qwen':
                prompt_text_formatted = f"<|im_start|>system\n{current_system_prompt}<|im_end|>\n<|im_start|>user\n{user_content_with_graph}<|im_end|>\n<|im_start|>assistant\n"
            elif llm_type == 'codellama':
                prompt_text_formatted = f"<s>[INST] <<SYS>>\n{current_system_prompt}\n<</SYS>>\n\n{user_content_with_graph} [/INST]"
            elif llm_type == 'deepseek':
                instruction_body = f"### Instruction:\n{user_content_with_graph}"
                instruction = f"{current_system_prompt}\n{instruction_body}"
                prompt_text_formatted = f"<｜begin▁of▁sentence｜>{instruction}\n### Response:\n"
            else: # Default/Auto format
                if current_system_prompt:
                    prompt_text_formatted = f"{current_system_prompt}\n\n{user_content_with_graph}\n\n"
                else:
                    prompt_text_formatted = f"{user_content_with_graph}\n\n"
            
            prompt_token_lengths.append(len(tokenizer.encode(prompt_text_formatted, add_special_tokens=False)))

            if answer is not None: # Training
                ans = answer[i] # answer is expected to be a list of strings
                ans_with_end = ans 
                if llm_type == 'qwen':
                    ans_with_end = ans + "<|im_end|>"
                elif llm_type == 'codellama':
                    ans_with_end = ans + " </s>"
                elif llm_type == 'deepseek':
                    ans_with_end = ans + "<|EOT|>"
                full_input_texts.append(prompt_text_formatted + ans_with_end)
            else: # Inference
                full_input_texts.append(prompt_text_formatted)

        batch_encoded = tokenizer(
            full_input_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
            add_special_tokens=False # Assumes chat templates are self-contained including BOS/EOS if model needs them per segment
        )

        input_ids = batch_encoded.input_ids.to(device)
        attention_mask = batch_encoded.attention_mask.to(device)
        
        labels = None
        if answer is not None:
            labels = input_ids.clone()
            
            # 1. Vectorize: Mask all padding tokens
            labels[attention_mask == 0] = -100
            
            # 2. Vectorize: Mask all prompt tokens (in non-padding parts)
            prompt_token_lengths_tensor = torch.tensor(
                prompt_token_lengths, dtype=torch.long, device=device
            ).unsqueeze(1) # Shape: (bsz, 1)

            is_content_token = (attention_mask == 1) # Shape: (bsz, seq_len)
            token_rank_in_sequence = torch.cumsum(is_content_token.long(), dim=1) # Ranks are 1-based for content tokens
            
            # Mask for tokens that are part of the prompt
            is_prompt_token_mask = (token_rank_in_sequence <= prompt_token_lengths_tensor) & is_content_token
            
            labels[is_prompt_token_mask] = -100

        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
        
        for i in range(bsz):
            graph_positions = (input_ids[i] == self.graph_token_id).nonzero(as_tuple=True)[0]
            assert len(graph_positions) > 0, f"Sample {i}: No graph placeholders found in input_ids"
            assert len(graph_positions) == nvtoken, (
                f"Sample {i}: Found {len(graph_positions)} graph placeholders ('{graph_token_placeholder_str}'), expected {nvtoken}. "
                f"Input_ids (last 50): ...{input_ids[i][-50:].tolist()}... Graph_token_id: {self.graph_token_id}"
            )
            
            nvtoken_start_idx = graph_positions[0]
            # if nvtoken > 1: # Check contiguity if multiple graph tokens
            #     for k_token in range(1, nvtoken):
            #         assert graph_positions[k_token] == nvtoken_start_idx + k_token, (
            #             f"Sample {i}: Graph tokens ('{graph_token_placeholder_str}') are not contiguous. "
            #             f"Positions: {graph_positions.tolist()}"
            #         )
            
            inputs_embeds[i, nvtoken_start_idx : nvtoken_start_idx + nvtoken, :] = gtokens[i]

        return input_ids, labels, inputs_embeds, attention_mask, prompt_token_lengths
    
    def forward(self, samples):
        """
        Compute forward pass and loss
        Args:
            samples:
            - summarization: (idx, graph_embs, code, code_summary)
            - translation: (idx, graph_embs, src_code, tgt_code)
            - clone_detection: (idx, graph_embs1, graph_embs2, code1, code2, label)
        """
        # Get sample components
        idx_batch, graph_embs, code_text, text = samples

        model_dev = next(self.parameters()).device
        
        # Normalize graph embeddings (supports graph-level vector or node sequence)
        multimodal_embeds, multimodal_atts = self._normalize_graph_inputs(graph_embs, device=model_dev)
        if torch.isnan(multimodal_embeds).any() or torch.isinf(multimodal_embeds).any():
            logger.error(f"FORWARD_QFORMER_PREP: NaN/Inf detected in multimodal_embeds! CSV_Indices (first 5): {idx_batch.tolist()[:5] if isinstance(idx_batch, torch.Tensor) else 'N/A'}")
            
        # Initialize query tokens
        query_tokens = self.query_tokens.expand(multimodal_embeds.shape[0], -1, -1).to(model_dev)
        
        # Qformer processing
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=multimodal_embeds,
            encoder_attention_mask=multimodal_atts,
            return_dict=True,
        )
        qformer_hidden_state = query_output.last_hidden_state
        if torch.isnan(qformer_hidden_state).any() or torch.isinf(qformer_hidden_state).any():
            logger.error(f"FORWARD_QFORMER_OUT: NaN/Inf detected in Qformer output (last_hidden_state)! CSV_Indices (first 5): {idx_batch.tolist()[:5] if isinstance(idx_batch, torch.Tensor) else 'N/A'}")

        # Project to LLM space
        gtokens = self.llm_proj(qformer_hidden_state[:, :query_tokens.size(1), :])
        if torch.isnan(gtokens).any() or torch.isinf(gtokens).any():
            logger.error(f"FORWARD_LLM_PROJ: NaN/Inf detected in gtokens (after llm_proj)! CSV_Indices (first 5): {idx_batch.tolist()[:5] if isinstance(idx_batch, torch.Tensor) else 'N/A'}")

        # Prepare LLM input (instruction template processing is integrated into prepare_lm_input)
        input_ids, labels, inputs_embeds, attention_mask, prompt_token_lengths = self.prepare_lm_input(
            gtokens=gtokens,
            text_input=code_text,
            answer=text
        )
        if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
            logger.error(f"FORWARD_LLM_INPUT: NaN/Inf detected in inputs_embeds for LLM! CSV_Indices (first 5): {idx_batch.tolist()[:5] if isinstance(idx_batch, torch.Tensor) else 'N/A'}")
            # You might want to log problematic input_ids or text here too
            # for i_debug in range(len(idx_batch)):
            #     if torch.isnan(inputs_embeds[i_debug]).any() or torch.isinf(inputs_embeds[i_debug]).any():
            #         logger.error(f"    Problematic sample CSV_idx: {idx_batch[i_debug].item()}, code_text: {code_text[i_debug][:100]}..., text: {text[i_debug][:100]}...")

        # Calculate loss through LLM
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )
        
        if torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any():
            logger.error(f"FORWARD_LLM_LOGITS: NaN/Inf detected in LLM logits! CSV_Indices (first 5): {idx_batch.tolist()[:5] if isinstance(idx_batch, torch.Tensor) else 'N/A'}")

        raw_loss = outputs.loss
        if raw_loss is None: # Should not happen if labels are provided and not all ignored
             logger.error(f"FORWARD_LLM_LOSS: Loss is None! This is unexpected. CSV_Indices (first 5): {idx_batch.tolist()[:5] if isinstance(idx_batch, torch.Tensor) else 'N/A'}")
             # Fallback to a zero tensor to prevent crash, but this is an error condition
             raw_loss = torch.tensor(0.0, device=model_dev, requires_grad=True)
        elif torch.isnan(raw_loss).any() or torch.isinf(raw_loss).any():
            logger.error(f"FORWARD_LLM_LOSS: NaN/Inf detected in raw_loss from LLM! CSV_Indices (first 5): {idx_batch.tolist()[:5] if isinstance(idx_batch, torch.Tensor) else 'N/A'}")


        # Return loss, gtokens, and logits
        return {
            "loss": raw_loss,
            "gtokens": gtokens,
            "logits": outputs.logits
        }
        
    @torch.no_grad()
    def generate(
        self,
        samples,
        gen_params=None,
    ):

        # Get current device, instead of using self.device
        device = next(self.parameters()).device
        
        # Extract sample components
        graph_embs, code_text = samples
        graph_embeds, graph_atts = self._normalize_graph_inputs(graph_embs, device=device)
        bs = graph_embeds.shape[0]
        
        with torch.no_grad():
            multimodal_embeds = graph_embeds
            multimodal_atts = graph_atts
            
            # Initialize query tokens, using the device of graph_embs
            query_tokens = self.query_tokens.expand(multimodal_embeds.shape[0], -1, -1).to(device)
            
            # Qformer processing
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=multimodal_embeds,
                encoder_attention_mask=multimodal_atts,
                return_dict=True,
            )
            
            # Project to LLM space
            gtokens = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        
            # Prepare input for generation
            input_ids, _, inputs_embeds, attention_mask, prompt_token_lengths = self.prepare_lm_input(
                gtokens=gtokens,
                text_input=code_text,
                answer=None
            )
            
            # logger.info(f"Starting generation, parameters: {gen_params}")
        
            # Generate text
            try:
                outputs = self.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    **gen_params
                )
    
                # Decode generated text
                response_output = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                return response_output
                
            except Exception as e:
                logger.error(f"Error during generation: {str(e)}", exc_info=True)
                return ["Generation failed: " + str(e)] * len(samples[0])
    
    
    @torch.no_grad()
    def generate_llm_only(
        self,
        text_inputs,
        gen_params=None,
    ):
        """
        Generate code summary using only LLM, without using graph representation
        Args:
            text_inputs: List of code texts
            gen_params: Dictionary of generation parameters
        Returns:
            List of generated code summaries
        """
        # Get current device
        device = next(self.parameters()).device
        
        # Get batch size
        bs = len(text_inputs)
        
        with torch.no_grad():
            # Get task configuration
            language = self.language
            task = self.task
            prompts_config = self.config.get("prompts", {})
            task_prompts = prompts_config.get(task, {})
            
            # Prepare system prompt and user prompt
            if task_prompts:
                system_prompt = task_prompts.get("system_prompt", "")
                user_prompt_template = task_prompts.get("user_prompt", "")
                system_prompt = system_prompt.format(language=language)
            
            # Determine LLM type
            llm_type = self.llm_type
            
            # Prepare formatted inputs
            formatted_inputs = []
            for code in text_inputs:
                # Format user prompt
                if task_prompts:
                    user_content = user_prompt_template.format(language=language, code=code)
                else:
                    user_content = code
                
                # Apply different formats based on model type
                if llm_type == 'qwen':
                    formatted_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
                elif llm_type == 'codellama':
                    formatted_text = f"<s>[INST] <<SYS>>{system_prompt}\n<</SYS>>\n\n{user_content} [/INST]"
                elif llm_type == 'deepseek':
                    instruction = f"{system_prompt}\n### Instruction:\n{user_content}" if system_prompt else f"### Instruction:\n{user_content}"
                    formatted_text = f"<｜begin▁of▁sentence｜>{instruction}\n### Response:\n"
                else:
                    # Default format
                    formatted_text = f"{system_prompt}\n\n{user_content}\n\n"
                
                formatted_inputs.append(formatted_text)
            
            # Batch encode inputs
            tokenizer = self.llm_tokenizer
            inputs = tokenizer(
                formatted_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_txt_len,
            ).to(device)
            # inputs_ids_lenths=(inputs.attention_mask).sum(1).tolist()
            inputs_ids_lenths = inputs.input_ids.shape[1]
            inputs_embeds = self.llm_model.get_input_embeddings()(inputs.input_ids)
            # Generate text
            try:
                outputs = self.llm_model.generate(
                    # input_ids=inputs.input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    **gen_params,
                )
                response_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # generated_ids_batch = outputs[:, inputs_ids_lenths:]
                # response_output = tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)
                
                # # Decode generated text
                # response_output = []
                # for i in range(bs):
                #     outputs_i = outputs.tolist()[i][inputs_ids_lenths:]
                #     response = tokenizer.decode(outputs_i, skip_special_tokens=True)
                    
                #     # Post-process
                #     # response = self._post_process_generated_text(response)
                #     response_output.append(response)
                
                return response_output
                
            except Exception as e:
                logger.error(f"Error during generation: {str(e)}", exc_info=True)
                return ["Generation failed: " + str(e)] * bs
    
    @torch.no_grad()
    def extract_attention_weights(
        self,
        samples,
        layer_idx=-1,  # Default to using the last layer
        head_idx=None,  # None means average all heads
    ):
        """
        Extract LLM attention weights for graph token
        Args:
            samples: (graph_embs, code_text) 
            layer_idx: Index of the layer to analyze, -1 means the last layer
            head_idx: Index of the attention head to analyze, None means average all heads
        Returns:
            dict: Dictionary containing attention weights and related information
        """
        device = next(self.parameters()).device
        
        # Extract sample components
        graph_embs, code_text = samples
        
        #  Normalize and ensure graph_embs are on the correct device
        multimodal_embeds, multimodal_atts = self._normalize_graph_inputs(graph_embs, device=device)
        
        #  Critical fix: Ensure query_tokens are on the correct device
        query_tokens = self.query_tokens.expand(multimodal_embeds.shape[0], -1, -1).to(device)
        
        # Qformer processing
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=multimodal_embeds,
            encoder_attention_mask=multimodal_atts,
            return_dict=True,
        )
        
        # Project to LLM space
        gtokens = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        
        # Prepare LLM input
        input_ids, _, inputs_embeds, attention_mask, prompt_token_lengths = self.prepare_lm_input(
            gtokens=gtokens,
            text_input=code_text,
            answer=None
        )
        
        # Get LLM attention weights
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        
        # Extract attention weights for the specified layer
        attentions = outputs.attentions[layer_idx]  # [batch_size, num_heads, seq_len, seq_len]
        
        # If a head is specified, select that head; otherwise average all heads
        if head_idx is not None:
            attention_weights = attentions[:, head_idx, :, :]  # [batch_size, seq_len, seq_len]
        else:
            attention_weights = attentions.mean(dim=1)  # [batch_size, seq_len, seq_len]
        
        # Find positions of graph tokens
        graph_token_positions = []
        nvtoken = gtokens.size(1)
        
        for i in range(input_ids.size(0)):
            positions = (input_ids[i] == self.graph_token_id).nonzero(as_tuple=True)[0]
            graph_token_positions.append(positions.cpu().tolist())
        
        return {
            "attention_weights": attention_weights.cpu(),
            "input_ids": input_ids.cpu(),
            "graph_token_positions": graph_token_positions,
            "tokenized_text": [self.llm_tokenizer.convert_ids_to_tokens(ids) for ids in input_ids.cpu()],
            "attention_mask": attention_mask.cpu(),
            "num_query_tokens": nvtoken
        }

    def visualize_graph_attention(
        self,
        attention_data,
        sample_idx=0,
        save_path=None,
        title=None
    ):
        """
        Visualize the attention heatmap for graph token
        Args:
            attention_data: Data returned from extract_attention_weights
            sample_idx: Index of the sample to visualize
            save_path: Save path
            title: Image title
        """
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import seaborn as sns
        import numpy as np
        
        # Set font for Chinese characters
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        attention_weights = attention_data["attention_weights"][sample_idx]  # [seq_len, seq_len]
        tokens = attention_data["tokenized_text"][sample_idx]
        graph_positions = attention_data["graph_token_positions"][sample_idx]
        attention_mask = attention_data["attention_mask"][sample_idx]
        
        # Only consider valid token positions
        valid_length = attention_mask.sum().item()
        attention_weights = attention_weights[:valid_length, :valid_length]
        tokens = tokens[:valid_length]
        
        # Extract attention to graph token
        graph_attention = attention_weights[:, graph_positions].mean(dim=1)  # Average attention to all graph tokens
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Upper plot: Complete attention matrix
        im1 = ax1.imshow(attention_weights.numpy(), cmap='Blues', aspect='auto')
        ax1.set_title(f'Complete Attention Matrix (Sample {sample_idx})', fontsize=14, pad=20)
        ax1.set_xlabel('Key Positions (Tokens being attended to)', fontsize=12)
        ax1.set_ylabel('Query Positions (Tokens attending)', fontsize=12)
        
        # Mark graph token positions
        for pos in graph_positions:
            ax1.axvline(x=pos, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax1.axhline(y=pos, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add colorbar
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Lower plot: Average attention to graph token
        ax2.bar(range(len(graph_attention)), graph_attention.numpy(), color='skyblue', alpha=0.7)
        ax2.set_title(f'Average Attention to Graph Token (Sample {sample_idx})', fontsize=14, pad=20)
        ax2.set_xlabel('Token Positions', fontsize=12)
        ax2.set_ylabel('Attention Weights', fontsize=12)
        
        # Mark graph token positions
        for pos in graph_positions:
            ax2.axvline(x=pos, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                       label='Graph Token' if pos == graph_positions[0] else "")
        
        if graph_positions:
            ax2.legend()
        
        # Set x-axis ticks (show part of tokens)
        step = max(1, len(tokens) // 20)  # Show at most 20 token labels
        tick_positions = list(range(0, len(tokens), step))
        tick_labels = [tokens[i][:10] + ('...' if len(tokens[i]) > 10 else '') for i in tick_positions]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to: {save_path}")
        
        plt.show()
        
        # Print statistics
        print(f"\n=== Attention Statistics (Sample {sample_idx}) ===")
        print(f"Sequence Length: {len(tokens)}")
        print(f"Graph Token Positions: {graph_positions}")
        print(f"Number of Graph Tokens: {len(graph_positions)}")
        
        if graph_positions:
            graph_self_attention = attention_weights[graph_positions, :][:, graph_positions].mean().item()
            other_to_graph_attention = graph_attention.mean().item()
            print(f"Average Self-Attention for Graph Token: {graph_self_attention:.4f}")
            print(f"Average Attention from Other Tokens to Graph Token: {other_to_graph_attention:.4f}")
            
            # Find the top 5 positions most attentive to graph token
            top_attention_indices = torch.topk(graph_attention, min(5, len(graph_attention))).indices
            print(f"\nTop Positions Attending to Graph Token:")
            for i, idx in enumerate(top_attention_indices):
                if idx.item() not in graph_positions:  # Exclude the graph token itself
                    token = tokens[idx.item()]
                    attention_val = graph_attention[idx.item()].item()
                    print(f"  Position {idx.item()}: '{token}' (Attention: {attention_val:.4f})")

    @classmethod
    def from_config(cls, cfg):
        """Create model instance from configuration"""
        model = cls(config=cfg)
        return model


    @torch.no_grad()
    def extract_comprehensive_attention_data(
        self,
        samples,
        graph_token_positions=None,
        analyze_all_layers=True,
        trim_before_graph=True,
        generation_mode=False,
        max_new_tokens=50,
        generation_analysis_steps="last_5"
    ):
        """
        Extract comprehensive attention data - pure data processing, without visualization
        
        Args:
            samples: (graph_embs, code_text)
            graph_token_positions: [start, end] Fixed positions, None means dynamic detection
            analyze_all_layers: Specify which layers to analyze
                - True: Analyze all layers (0 to num_layers-1)
                - False: Analyze the last 3 layers
                - list: Specify layer index list, e.g., [0, 5, 10, 15, 20, 27] 
            trim_before_graph: Whether to trim the part before graph token
            generation_mode: Whether to analyze during generation phase
            max_new_tokens: Maximum new token count during generation mode
            generation_analysis_steps: "all", "last_5", "first_5", or a number
        
        Returns:
            dict: Dictionary containing all attention data and statistics
        """
        device = next(self.parameters()).device
        graph_embs, code_text = samples
        graph_embeds, graph_atts = self._normalize_graph_inputs(graph_embs, device=device)
        
        batch_size = len(code_text)
        num_layers = self.llm_model.config.num_hidden_layers
        
        print(f"Extracting attention data ({'generation mode' if generation_mode else 'prompt mode'})")
        print(f"Batch: {batch_size}, Layers: {num_layers}, Device: {device}")
        
        # Data preprocessing
        multimodal_embeds = graph_embeds
        multimodal_atts = graph_atts
        query_tokens = self.query_tokens.expand(multimodal_embeds.shape[0], -1, -1).to(device)
        
        # Q-Former processing
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=multimodal_embeds,
            encoder_attention_mask=multimodal_atts,
            return_dict=True,
        )
        
        gtokens = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        
        if generation_mode:
            return self._extract_generation_attention_data(
                gtokens, code_text, max_new_tokens, generation_analysis_steps, 
                analyze_all_layers, graph_token_positions, trim_before_graph
            )
        else:
            return self._extract_prompt_attention_data(
                gtokens, code_text, analyze_all_layers, 
                graph_token_positions, trim_before_graph, batch_size
            )

    def _extract_generation_attention_data(
        self, gtokens, code_text, max_new_tokens, analysis_steps, 
        analyze_all_layers, graph_token_positions, trim_before_graph
    ):
        """Extract attention data during generation phase"""
        
        # Prepare initial input
        input_ids, _, inputs_embeds, attention_mask, _ = self.prepare_lm_input(
            gtokens=gtokens, text_input=code_text, answer=None
        )
        
        print("Starting generation process...")
        
        # Generation process, saving attention
        generation_outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=True,
            do_sample=False,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
        )
        
        step_attentions = generation_outputs.attentions
        generated_sequences = generation_outputs.sequences
        
        print(f"Generated {len(step_attentions)} steps")
        
        # Determine Graph Token positions
        graph_start, graph_end = self._get_graph_token_positions(
            input_ids, graph_token_positions
        )
        
        # Determine analysis steps
        steps_to_analyze = self._get_analysis_steps(step_attentions, analysis_steps)
        
        # Determine analysis layers
        layers_to_analyze = self._get_analysis_layers(
            len(step_attentions[0]), analyze_all_layers
        )
        
        print(f"Analyzing steps: {len(steps_to_analyze)}, layers: {len(layers_to_analyze)}")
        
        # Extract attention for each step and layer
        attention_matrix = []  # [num_layers, num_steps]
        
        for layer_idx in layers_to_analyze:
            layer_step_results = []
            
            for step_idx in steps_to_analyze:
                # Get attention for the current step and layer
                step_layer_attention = step_attentions[step_idx][layer_idx]
                # Shape: [batch_size, num_heads, current_seq_len, current_seq_len]
                
                # Analyze attention of the newly generated token (last position) to graph tokens
                new_token_attention = step_layer_attention[:, :, -1, :]  # Attention for the last token
                
                current_seq_len = step_layer_attention.shape[-1]
                
                if graph_end <= current_seq_len:
                    # Extract attention to graph tokens
                    new_token_to_graph = new_token_attention[:, :, graph_start:graph_end]
                    # Sum attention to graph tokens, averaging across heads and batches
                    graph_attention = new_token_to_graph.sum(dim=-1).mean(dim=1).mean(dim=0)
                    layer_step_results.append(graph_attention.float().cpu().item())
                else:
                    layer_step_results.append(0.0)
            
            attention_matrix.append(layer_step_results)
        
        # Convert to tensor
        heatmap_data = torch.tensor(attention_matrix, dtype=torch.float32)
        
        # Decode generated text
        original_prompt_length = inputs_embeds.shape[1]
        generated_texts = []
        
        for batch_idx in range(generated_sequences.shape[0]):
            generated_tokens = generated_sequences[batch_idx][original_prompt_length:]
            generated_text = self.llm_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        print(f"Generation mode data extraction completed, shape: {heatmap_data.shape}")
        
        return {
            "heatmap_data": heatmap_data,
            "analysis_mode": "generation",
            "num_layers_analyzed": len(layers_to_analyze),
            "total_layers": len(step_attentions[0]),
            "num_generation_steps": len(steps_to_analyze),
            "total_generation_steps": len(step_attentions),
            "generation_steps_analyzed": steps_to_analyze,
            "layers_analyzed": layers_to_analyze,
            "graph_token_range": [graph_start, graph_end],
            "graph_token_count": graph_end - graph_start,
            "batch_size": generated_sequences.shape[0],
            "max_attention": heatmap_data.max().item(),
            "mean_attention": heatmap_data.mean().item(),
            "attention_std": heatmap_data.std().item(),
            "generated_texts": generated_texts,
            "original_prompt_length": original_prompt_length,
            "analysis_steps_config": analysis_steps
        }

    def _extract_prompt_attention_data(
        self, gtokens, code_text, analyze_all_layers, 
        graph_token_positions, trim_before_graph, batch_size
    ):
        """Extract attention data during prompt phase"""
        
        # Prepare LLM input
        input_ids, _, inputs_embeds, attention_mask, _ = self.prepare_lm_input(
            gtokens=gtokens, text_input=code_text, answer=None
        )
        
        # Get attention weights
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        
        # Determine Graph Token positions
        graph_start, graph_end = self._get_graph_token_positions(
            input_ids, graph_token_positions
        )
        
        #  Validate number of graph tokens
        graph_token_count = graph_end - graph_start
        expected_graph_count = gtokens.shape[1]  # Should be 32
        
        print(f" Graph Token validation:")
        print(f"  Detected Graph Token positions: [{graph_start}:{graph_end}]")
        print(f"  Number of Graph Tokens: {graph_token_count}")
        print(f"  Expected Number of Graph Tokens: {expected_graph_count}")
        
        if graph_token_count != expected_graph_count:
            print(f" Warning: Graph Token count mismatch! Detected {graph_token_count}, expected {expected_graph_count}")
        
        # Determine analysis layers
        layers_to_analyze = self._get_analysis_layers(
            len(outputs.attentions), analyze_all_layers
        )
        
        # Get valid sequence lengths
        valid_lengths = attention_mask.sum(dim=1).cpu()
        max_valid_length = valid_lengths.max().item()
        
        print(f"Sample valid lengths: {valid_lengths.tolist()}")
        
        # Extract attention for each layer
        all_layers_results = []
        device = next(self.parameters()).device
        
        for layer_idx in layers_to_analyze:
            attn_tensor = outputs.attentions[layer_idx]
            
            # Extract attention to Graph Token
            attn_to_graph = attn_tensor[:, :, :, graph_start:graph_end]
            sum_attn_to_graph_per_head = attn_to_graph.sum(dim=-1)
            avg_attn_to_graph_per_query = sum_attn_to_graph_per_head.mean(dim=1)
            
            # Correct batch averaging (considering padding)
            layer_result = torch.zeros(max_valid_length, device=device)
            valid_counts = torch.zeros(max_valid_length, device=device)
            
            for batch_idx in range(batch_size):
                valid_len = valid_lengths[batch_idx].item()
                layer_result[:valid_len] += avg_attn_to_graph_per_query[batch_idx, :valid_len]
                valid_counts[:valid_len] += 1
            
            valid_counts = torch.clamp(valid_counts, min=1)
            layer_result = layer_result / valid_counts
            
            #  Improved trimming logic to ensure correctness
            if trim_before_graph:
                vis_start = graph_start
                if vis_start < len(layer_result):
                    layer_result = layer_result[vis_start:]
                    
                    #  Validate trimmed sequence
                    remaining_length = len(layer_result)
                    print(f"  Trimmed sequence length: {remaining_length}")
                    print(f"  The first {min(graph_token_count, remaining_length)} positions should be Graph Tokens")
                else:
                    print(f" Warning: Trim start position {vis_start} exceeds sequence length {len(layer_result)}")
                    vis_start = 0
            else:
                vis_start = 0
            
            all_layers_results.append(layer_result.float().cpu())
        
        # Stack results
        heatmap_data = torch.stack(all_layers_results)
        
        print(f" Prompt mode data extraction completed, shape: {heatmap_data.shape}")
        
        seq_len = heatmap_data.shape[1]
        
        #  Enhanced return information, including more validation data
        return {
            "heatmap_data": heatmap_data,
            "analysis_mode": "prompt",
            "num_layers_analyzed": len(layers_to_analyze),
            "total_layers": len(outputs.attentions),
            "seq_length": seq_len,
            "original_seq_length": max_valid_length,
            "trimmed_start_pos": vis_start,
            "batch_size": batch_size,
            "valid_lengths": valid_lengths.tolist(),
            "graph_token_range": [graph_start, graph_end],
            "graph_token_count": graph_token_count,
            "expected_graph_count": expected_graph_count,
            "graph_tokens_verified": graph_token_count == expected_graph_count,
            "max_attention": heatmap_data.max().item(),
            "mean_attention": heatmap_data.mean().item(),
            "attention_std": heatmap_data.std().item(),
            "layers_analyzed": layers_to_analyze,
            "trim_before_graph": trim_before_graph,
            "position_mapping": {
                "original_graph_start": graph_start,
                "original_graph_end": graph_end,
                "trimmed_graph_start": 0 if trim_before_graph else graph_start,
                "trimmed_graph_end": graph_token_count if trim_before_graph else graph_end
            }
        }

    def _get_graph_token_positions(self, input_ids, graph_token_positions):
        """Get Graph Token positions"""
        if graph_token_positions is not None:
            graph_start, graph_end = graph_token_positions
            print(f"Using fixed Graph Token positions: [{graph_start}:{graph_end}]")
        else:
            sample_0_positions = (input_ids[0] == self.graph_token_id).nonzero(as_tuple=True)[0]
            if len(sample_0_positions) > 0:
                graph_start = sample_0_positions[0].item()
                graph_end = sample_0_positions[-1].item() + 1
                print(f"Dynamic detection of Graph Token positions: [{graph_start}:{graph_end}]")
            else:
                raise ValueError("Graph Token position not found!")
        return graph_start, graph_end

    def _get_analysis_steps(self, step_attentions, analysis_steps):
        """Determine which generation steps to analyze"""
        if analysis_steps == "all":
            return list(range(len(step_attentions)))
        elif analysis_steps == "last_5":
            return list(range(max(0, len(step_attentions)-5), len(step_attentions)))
        elif analysis_steps == "first_5":
            return list(range(min(5, len(step_attentions))))
        elif isinstance(analysis_steps, int):
            return list(range(min(analysis_steps, len(step_attentions))))
        else:
            return list(range(min(5, len(step_attentions))))

    def _get_analysis_layers(self, total_layers, analyze_all_layers):
        """
        Determine which layers to analyze
        
        Args:
            total_layers: Total number of layers in the model
            analyze_all_layers: 
                - True: Analyze all layers
                - False: Analyze the last 3 layers
                - list: Specify which layers to analyze, e.g., [0, 10, 20, 27]
        
        Returns:
            list: List of layer indices to analyze
        """
        if analyze_all_layers is True:
            return list(range(total_layers))
        elif isinstance(analyze_all_layers, list):
            #  Validate and filter valid layer indices
            valid_layers = []
            for layer_idx in analyze_all_layers:
                if isinstance(layer_idx, int) and 0 <= layer_idx < total_layers:
                    valid_layers.append(layer_idx)
                else:
                    print(f" Warning: Layer index {layer_idx} is invalid (Total layers: {total_layers}), skipping")
            
            if not valid_layers:
                print(f" Error: No valid layer indices, falling back to the last 3 layers")
                return [total_layers-3, total_layers-2, total_layers-1]
            
            #  Sort and deduplicate by index
            valid_layers = sorted(list(set(valid_layers)))
            print(f" Analyzing specified {len(valid_layers)} layers: {valid_layers}")
            return valid_layers
        else:
            # Default to analyzing the last 3 layers
            return [total_layers-3, total_layers-2, total_layers-1]
        
class CodeSummaryDataset(torch.utils.data.Dataset):
    """Code summarization dataset"""
    
    def __init__(self, csv_path, max_length=512):
        """
        Initialize dataset
        Args:
            csv_path: CSV file path
            max_length: Maximum sequence length
        """
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length
        self.has_node_embs = 'node_embs' in self.df.columns
        
        # Check if necessary columns exist
        required_columns = {'idx'}
        if not self.has_node_embs:
            required_columns.add('graph_emb')
        possible_code_columns = {'code', 'source_code', 'input_code'}
        possible_summary_columns = {'answer', 'code_summary', 'summary', 'description'}
        
        missing_required = required_columns - set(self.df.columns)
        if missing_required:
            raise ValueError(f"CSV is missing required columns: {missing_required}")
        
        # Determine code column
        self.code_column = None
        for col in possible_code_columns:
            if col in self.df.columns:
                self.code_column = col
                break
        if not self.code_column:
            raise ValueError(f"Cannot find code column in CSV, please check if one of the following column names exists: {possible_code_columns}")
        
        # Determine summary column
        self.summary_column = None
        for col in possible_summary_columns:
            if col in self.df.columns:
                self.summary_column = col
                break
        if not self.summary_column:
            raise ValueError(f"Cannot find summary column in CSV, please check if one of the following column names exists: {possible_summary_columns}")
            
        logging.info(f"Code summarization dataset loaded successfully, using code column: '{self.code_column}', summary column: '{self.summary_column}'")
        logging.info(f"Dataset size: {len(self.df)} entries")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get dataset item
        Returns: (idx, graph_emb, code, code_summary)
        """
        item = self.df.iloc[idx]
        
        try:
            # Parse graph embedding (graph-level vector or per-node list)
            if self.has_node_embs:
                node_embs = json.loads(item['node_embs'])
                graph_emb = torch.tensor(node_embs, dtype=torch.float)
            else:
                graph_emb_str = item['graph_emb'].strip('[]').split(',')
                graph_emb = torch.tensor([float(x) for x in graph_emb_str], dtype=torch.float)
            
            # Get code text
            code = str(item[self.code_column]) 
            
            # Get code summary
            code_summary = str(item[self.summary_column])
            
        except Exception as e:
             logging.error(f"Error processing sample {idx} (CSV index: {item.get('idx', 'N/A')}): {e}")
             raise ValueError(f"Error processing sample {idx}") from e

        return item['idx'], graph_emb, code, code_summary
    
class CodeTranslateDataset(torch.utils.data.Dataset):
    """Code translation dataset"""
    
    def __init__(self, csv_path, max_length=512):
        """
        Initialize dataset
        Args:
            csv_path: CSV file path
            max_length: Maximum sequence length
        """
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length
        self.has_node_embs = 'node_embs' in self.df.columns
        
        # Check if necessary columns exist
        required_columns = {'idx', 'src_code', 'tgt_code'}
        if not self.has_node_embs:
            required_columns.add('graph_emb')
        missing_required = required_columns - set(self.df.columns)
        if missing_required:
            raise ValueError(f"CSV is missing required columns: {missing_required}")
            
        logging.info(f"Code translation dataset loaded successfully, dataset size: {len(self.df)} entries")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get dataset item
        Returns: (idx, graph_emb, src_code, tgt_code)
        """
        item = self.df.iloc[idx]
        
        try:
            # Parse graph embedding
            if self.has_node_embs:
                node_embs = json.loads(item['node_embs'])
                graph_emb = torch.tensor(node_embs, dtype=torch.float)
            else:
                graph_emb_str = item['graph_emb'].strip('[]').split(',')
                graph_emb = torch.tensor([float(x) for x in graph_emb_str], dtype=torch.float)
            
            # Get source code text
            src_code = str(item['src_code']) 
            tgt_code = str(item['tgt_code']) 
            
        except Exception as e:
             logging.error(f"Error processing sample {idx} (CSV index: {item.get('idx', 'N/A')}): {e}")
             raise ValueError(f"Error processing sample {idx}") from e

        return item['idx'], graph_emb, src_code, tgt_code
