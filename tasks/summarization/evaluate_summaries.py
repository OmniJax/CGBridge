import os
import numpy as np
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from sentence_transformers import SentenceTransformer
import nltk
from tqdm import tqdm
import argparse
import sys
import pandas as pd
import bert_score  

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4')
"""


python evaluate_summaries.py --input /path/to/test_summary.csv

"""

SENTENCE_BERT_MODEL = "/path/to/models/all-MiniLM-L6-v2"

BERTSCORE_MODEL = "/path/to/models/roberta-large" 

def load_embedding_model():
    # Load the Sentence BERT model
    model = SentenceTransformer(SENTENCE_BERT_MODEL)
    return model

def get_embeddings(texts, model):
    # Get embeddings for the given texts
    if not texts:
        return []
    
    embeddings = model.encode(texts)
    return embeddings

def calculate_bleu(reference, candidate):
    # Calculate BLEU score between reference and candidate
    reference = str(reference) if reference is not None else ""
    candidate = str(candidate) if candidate is not None else ""
    
    if not reference or not candidate:
        return 0.0
    
    try:
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()
        
        if not reference_tokens or not candidate_tokens:
            return 0.0
        
        smoothing = SmoothingFunction().method1
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
    except Exception as e:
        print(f"BLEU calculation error: {str(e)}")
        return 0.0

def calculate_meteor(reference, candidate):
    # Calculate METEOR score between reference and candidate
    reference = str(reference) if reference is not None else ""
    candidate = str(candidate) if candidate is not None else ""
    
    if not reference or not candidate:
        return 0.0
    
    try:
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()
        
        if not reference_tokens or not candidate_tokens:
            return 0.0
        
        return meteor_score([reference_tokens], candidate_tokens)
    except Exception as e:
        print(f"METEOR calculation error: {str(e)}")
        return 0.0

def calculate_rouge_l(reference, candidate):
    # Calculate ROUGE-L score between reference and candidate
    reference = str(reference) if reference is not None else ""
    candidate = str(candidate) if candidate is not None else ""
    
    if not reference or not candidate:
        return 0.0
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(candidate, reference)[0]
        return scores['rouge-l']['f']
    except Exception as e:
        print(f"ROUGE-L calculation error: {str(e)}")
        return 0.0

def calculate_cosine_similarity(reference_embedding, candidate_embedding):
    # Calculate cosine similarity between two embeddings
    if len(reference_embedding) == 0 or len(candidate_embedding) == 0:
        return 0.0
    
    similarity = np.dot(reference_embedding, candidate_embedding) / (np.linalg.norm(reference_embedding) * np.linalg.norm(candidate_embedding))
    return float(similarity)  

def calculate_bertscore(references, candidates, model_path, lang='en'):
    # Calculate BERTScore for the given references and candidates
    references = [str(ref) if ref is not None else "" for ref in references]
    candidates = [str(cand) if cand is not None else "" for cand in candidates]
    
    valid_pairs = [(ref, cand) for ref, cand in zip(references, candidates) if ref and cand]
    
    if not valid_pairs:
        return [0.0] * len(references)
    
    valid_refs, valid_cands = zip(*valid_pairs)
    
    try:
        P, R, F1 = bert_score.score(valid_cands, valid_refs, model_type=model_path, num_layers=17, lang=lang, verbose=False)
        
        F1_scores = [float(score) for score in F1]
        
        results = []
        pair_idx = 0
        for ref, cand in zip(references, candidates):
            if ref and cand:
                results.append(F1_scores[pair_idx])
                pair_idx += 1
            else:
                results.append(0.0)
        
        return results
    except Exception as e:
        print(f"BERTScore calculation error (skipping): {str(e)}")
        print("⚠️  BERTScore calculation failed, returning 0 score. Please check if the model path is correct or if there are network issues.")
        return [0.0] * len(references)

# Read results from CSV file
def read_results(file_path):
    try:
        df = pd.read_csv(file_path)
        
        required_columns = ['idx', 'code', 'reference', 'generated']
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: Missing '{col}' column in CSV file")
                if len(df.columns) >= 4:
                    df.columns = required_columns[:len(df.columns)]
        
        return df
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return pd.DataFrame()

def evaluate_results(results_file):
    # Evaluate results from the given file
    df = read_results(results_file)
    
    if df.empty:
        print(f"No results found: {results_file}")
        return None, None
    
    print(f"Evaluating {len(df)} samples...")
    
    model = load_embedding_model()
    
    df['bleu'] = 0.0
    df['meteor'] = 0.0
    df['rouge_l'] = 0.0
    df['sentence_bert'] = 0.0
    df['bertscore'] = 0.0  
    
    references = list(df['reference'])
    candidates = list(df['generated'])
    
    print("Calculating BERTScore...")
    bertscore_results = calculate_bertscore(references, candidates, model_path=BERTSCORE_MODEL)
    df['bertscore'] = bertscore_results
    
    print("Calculating other metrics...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        reference = row['reference']
        candidate = row['generated']
        
        bleu_score = calculate_bleu(reference, candidate)
        df.at[idx, 'bleu'] = bleu_score
        
        meteor_score_value = calculate_meteor(reference, candidate)
        df.at[idx, 'meteor'] = meteor_score_value
        
        rouge_l_score = calculate_rouge_l(reference, candidate)
        df.at[idx, 'rouge_l'] = rouge_l_score
        
        if reference is not None and candidate is not None:
            reference_str = str(reference)
            candidate_str = str(candidate)
            
            if reference_str and candidate_str:
                reference_embedding = get_embeddings([reference_str], model)[0]
                candidate_embedding = get_embeddings([candidate_str], model)[0]
                similarity = calculate_cosine_similarity(reference_embedding, candidate_embedding)
                df.at[idx, 'sentence_bert'] = similarity
    
    avg_metrics_df = pd.DataFrame({
        'metric': ['bleu', 'meteor', 'rouge_l', 'sentence_bert', 'bertscore'],  
        'value': [
            df['bleu'].mean(),
            df['meteor'].mean(),
            df['rouge_l'].mean(),
            df['sentence_bert'].mean(),
            df['bertscore'].mean()  
        ]
    })
    
    results_df = df[['idx', 'reference', 'generated', 'bleu', 'meteor', 'rouge_l', 'sentence_bert', 'bertscore']].copy()  
    results_df.rename(columns={
        'generated': 'candidate'
    }, inplace=True)
    
    return avg_metrics_df, results_df

def save_evaluation_result(avg_metrics_df, results_df, input_file):
    """Save evaluation results to CSV file"""
    input_dir = os.path.dirname(input_file)
    if not input_dir:  # If it's a relative path with no directory part
        input_dir = '.'
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    avg_metrics_file = os.path.join(input_dir, f"{base_name}_avg_metrics.csv")
    detailed_results_file = os.path.join(input_dir, f"{base_name}_detailed_results.csv")
    
    # Save results
    avg_metrics_df.to_csv(avg_metrics_file, index=False)
    results_df.to_csv(detailed_results_file, index=False)
    
    print(f"Average metrics saved to {avg_metrics_file}")
    print(f"Detailed results saved to {detailed_results_file}")
    
    # Print summary of results
    print("\n=== Evaluation Results Summary ===")
    for _, row in avg_metrics_df.iterrows():
        print(f"{row['metric']}: {row['value']:.4f}")

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate code summary generation results')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input CSV file')
    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist - {args.input}")
        sys.exit(1)
    
    # Evaluate results
    avg_metrics_df, results_df = evaluate_results(args.input)
    
    # If evaluation is successful, save results
    if avg_metrics_df is not None and results_df is not None:
        save_evaluation_result(avg_metrics_df, results_df, args.input)
    else:
        print("Evaluation failed, no results generated.")
        sys.exit(1)

if __name__ == "__main__":
    main() 