import os
import torch
import json
from transformers import RobertaTokenizer, RobertaModel

def calculate_edge_embeddings(model_path, output_path, device="cuda:0"):
    """Pre-calculate all edge type embeddings and save to local"""
    
    # Initialize model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path).to(device)
    
    # All possible edge types
    edge_texts = [
        # CFG edge types 17
        'sequential execution',
        'true branch',
        'false branch',
        'alternate condition branch',
        'condition evaluation',
        'for loop body',
        'for loop iteration range',
        'while loop body',
        'while loop condition',
        'try block',
        'exception handler',
        'finally block',
        'block exit',
        'loop exit',
        'loop back',
        'break jump',
        'condition false jump',
        # DFG edge types 2
        'contributes to',
        'flows to',
        # AST edge types 11
        'has name',
        'has parameters',
        'has body',
        'has condition',
        'has then body',
        'has else body',
        'has elif branch',
        'has target',
        'has value',
        'contains',
        'function call'    ]
    
    # Calculate embeddings for each edge type
    edge_embedding_dict = {}
    
    # Batch encode all edge texts
    inputs = tokenizer(edge_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
    
    # Store embeddings in dictionary
    for i, edge_text in enumerate(edge_texts):
        edge_embedding_dict[edge_text] =embeddings[i]
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save(edge_embedding_dict, output_path + ".pt")
    
    print(f"Calculated and saved embeddings for {len(edge_embedding_dict)} edge types")
    print(f"PyTorch format saved to: {output_path}.pt")
    
    return edge_embedding_dict

# Example usage
if __name__ == "__main__":
    model_name='unixcoder'
    model_path = f"/path/to/models/{model_name}-base"
    output_path = f"/path/to/graph_datasets/enc_by_{model_name}/{model_name}_edge_embeddings_cache.pt"
    
    
    edge_embedding_dict = calculate_edge_embeddings(
        model_path=model_path,
        output_path=output_path,
        device="cuda"  # Can be modified as needed
    )
    
    print('edge_embedding_dict DONE')
    
    