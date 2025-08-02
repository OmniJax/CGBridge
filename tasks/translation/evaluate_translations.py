import os
import argparse
import sys
import pandas as pd
from codebleu import calc_codebleu 
from tqdm import tqdm

import re

def extract_java_code_from_llm_output(raw_llm_output: str) -> str:
    """
    Extracts the first Java code block from the LLM's raw output.
    Enhanced to handle more formats and filter out invalid outputs.
    """
    if not isinstance(raw_llm_output, str):
        return ""

    original_stripped_output = raw_llm_output.strip()

    invalid_patterns = [
        r"^\[PYTHON\]",  
        r"^\[/INST\]",   
        r"^(\[/INST\]\s*){3,}",  
        r"^Expected Output:",   
        r"^(\[PYTHON\]\s*){2,}", 
        r"^(\s*-->\s*){2,}",     
        r"^#\[/INST\]",          
        r"^\(\)\s*\[PYTHON\]",  
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, original_stripped_output, re.MULTILINE):
            print(f"Warning: Detected invalid output format, skipping processing: '{original_stripped_output[:100]}...'")
            return ""
    
    java_code_match = re.search(r"\*\*Java Code:\*\*\s*```(?:java)?\s*([\s\S]*?)```", original_stripped_output, re.DOTALL)
    if java_code_match:
        extracted = java_code_match.group(1).strip()
        return clean_extracted_code(extracted)
    
    match_java = re.search(r"```java\s*([\s\S]*?)```", original_stripped_output, re.DOTALL)
    if match_java:
        extracted = match_java.group(1).strip()
        return clean_extracted_code(extracted)

    match_generic = re.search(r"```([\s\S]*?)```", original_stripped_output, re.DOTALL)
    if match_generic:
        extracted = match_generic.group(1).strip()
        if looks_like_java_code(extracted):
            return clean_extracted_code(extracted)
    
    code_section_match = re.search(r"\*\*(?:Java\s+)?Code:\*\*\s*([\s\S]*?)(?=\*\*Note:|\*\*Explanation:|$)", original_stripped_output, re.DOTALL)
    if code_section_match:
        extracted = code_section_match.group(1).strip()
        if looks_like_java_code(extracted):
            return clean_extracted_code(extracted)
    
    return apply_traditional_cleaning(original_stripped_output)

def clean_extracted_code(code: str) -> str:
    code = re.sub(r"^(java|JAVA)\s*", "", code)
    
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = line.rstrip()
        if cleaned_line or (cleaned_lines and cleaned_lines[-1]):  
            cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines).strip()

def looks_like_java_code(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return False
    
    # Check for Java keywords
    java_indicators = [
        r'\bpublic\s+class\b',
        r'\bpublic\s+static\s+void\b',
        r'\bimport\s+java\.',
        r'\bSystem\.out\.print',
        r'\bpublic\s+static\b',
        r'\b(int|String|boolean|double|float)\s+\w+\s*[=;]',
        r'\bclass\s+\w+\s*\{',
    ]
    
    for indicator in java_indicators:
        if re.search(indicator, text):
            return True
    
    # Check basic code structure
    if '{' in text and '}' in text and ';' in text:
        return True
    
    return False

def apply_traditional_cleaning(text: str) -> str:
    common_intros = [
        "Here's the Java translation of the given Python code:",
        "Here is the Java equivalent of your Python code:",
        "Here is the Java code:",
        "Certainly, here is the Java code:",
        "The Java translation is as follows:",
        "**Java Code:**",
        "Java Code:",
    ]
    
    common_outros = [
        "Explanation:",
        "Note:",
        "**Note:",
        "This Java program does exactly the same thing",
        "This Java code performs the same operations",
        "Let me know if you have any other questions.",
    ]

    cleaned_output = text

    # Try to remove introductory phrases
    for intro in common_intros:
        if cleaned_output.startswith(intro):
            cleaned_output = cleaned_output[len(intro):].lstrip()
            break
    
    # Try to remove concluding phrases
    for outro in common_outros:
        if outro in cleaned_output:
            idx = cleaned_output.find(outro)
            potential_code = cleaned_output[:idx].rstrip()
            if looks_like_java_code(potential_code):
                cleaned_output = potential_code
                break
    
    return cleaned_output.strip()

# Calculate CodeBLEU score
def calculate_codebleu_score(reference, candidate, lang, weights=(0.25, 0.25, 0.25, 0.25)):

    reference = str(reference) if reference is not None else ""
    candidate = str(candidate) if candidate is not None else ""

    default_scores = {
        'codebleu': 0.0,
        'ngram_match_score': 0.0,
        'weighted_ngram_match_score': 0.0,
        'syntax_match_score': 0.0,
        'dataflow_match_score': 0.0
    }

    if not reference or not candidate:
        return default_scores

    try:
        
        result = calc_codebleu([[reference]], [candidate], lang, weights=weights)

        if isinstance(result, dict):
            return {
                'codebleu': float(result.get('codebleu', 0.0)),
                'ngram_match_score': float(result.get('ngram_match_score', 0.0)), # BLEU component
                'weighted_ngram_match_score': float(result.get('weighted_ngram_match_score', 0.0)),
                'syntax_match_score': float(result.get('syntax_match_score', 0.0)),
                'dataflow_match_score': float(result.get('dataflow_match_score', 0.0))
            }
        # Fallback for older versions or different return types, though less likely with current library versions
        elif isinstance(result, tuple) and len(result) > 0: 
            return {**default_scores, 'codebleu': float(result[0])} # Only codebleu, others 0
        else: 
            try:
                return {**default_scores, 'codebleu': float(result)} # Only codebleu, others 0
            except (TypeError, ValueError):
                print(f"Unexpected result format from calc_codebleu for candidate '{candidate[:50]}...': {result}")
                return default_scores

    except Exception as e:
        print(f"CodeBLEU calculation error for candidate '{candidate[:50]}...': {str(e)}")
        return default_scores

def read_translation_results(file_path):
    try:
        df = pd.read_csv(file_path)
        index_col = None
        if 'index' in df.columns:
            index_col = 'index'
        elif 'idx' in df.columns:
            index_col = 'idx'
        else:
            print(f"Error: CSV file missing required index column ('index' or 'idx')")
            return pd.DataFrame(), None
        
        required_columns = ['code', 'reference', 'generated']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: CSV file missing required column '{col}'")
                return pd.DataFrame(), None
                
        return df, index_col
    except Exception as e:
        print(f"Error reading CSV file '{file_path}': {str(e)}")
        return pd.DataFrame(), None

def evaluate_translations(results_file, lang, source_lang, task_type="translation"):
    df, index_col = read_translation_results(results_file)

    if df.empty or index_col is None:
        print(f"No data found or error reading file: {results_file}")
        return None, None

    print(f"Evaluating {len(df)} samples for CodeBLEU ({source_lang} to {lang})...")

    df['codebleu'] = 0.0
    df['ngram_match_score'] = 0.0
    df['weighted_ngram_match_score'] = 0.0
    df['syntax_match_score'] = 0.0
    df['dataflow_match_score'] = 0.0

    valid_outputs = 0
    invalid_outputs = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating CodeBLEU"):
        reference_code = row['reference']
        generated_code_raw = row['generated']
        
        generated_code = extract_java_code_from_llm_output(generated_code_raw)
        
        if not generated_code:
            invalid_outputs += 1
            df.at[idx, 'codebleu'] = 0.0
            df.at[idx, 'ngram_match_score'] = 0.0
            df.at[idx, 'weighted_ngram_match_score'] = 0.0
            df.at[idx, 'syntax_match_score'] = 0.0
            df.at[idx, 'dataflow_match_score'] = 0.0
            continue
        
        valid_outputs += 1
        scores = calculate_codebleu_score(reference_code, generated_code, lang)
        
        df.at[idx, 'codebleu'] = scores['codebleu']
        df.at[idx, 'ngram_match_score'] = scores['ngram_match_score']
        df.at[idx, 'weighted_ngram_match_score'] = scores['weighted_ngram_match_score']
        df.at[idx, 'syntax_match_score'] = scores['syntax_match_score']
        df.at[idx, 'dataflow_match_score'] = scores['dataflow_match_score']
    
    print(f"\nEvaluation Statistics:")
    print(f"Valid Outputs: {valid_outputs}/{len(df)} ({valid_outputs/len(df)*100:.1f}%)")
    print(f"Invalid Outputs: {invalid_outputs}/{len(df)} ({invalid_outputs/len(df)*100:.1f}%)")
    
    avg_metrics_df = pd.DataFrame({
        'metric': ['codebleu', 'ngram_match_score', 'weighted_ngram_match_score', 'syntax_match_score', 'dataflow_match_score'],
        'value': [
            df['codebleu'].mean(),
            df['ngram_match_score'].mean(),
            df['weighted_ngram_match_score'].mean(),
            df['syntax_match_score'].mean(),
            df['dataflow_match_score'].mean()
        ]
    })
    
    detailed_results_df = df[[index_col, 'reference', 'generated', 'codebleu', 'ngram_match_score', 'weighted_ngram_match_score', 'syntax_match_score', 'dataflow_match_score']].copy()
    
    return avg_metrics_df, detailed_results_df

def save_evaluation_results(avg_metrics_df, detailed_results_df, input_file):
    """Saves evaluation results to CSV files."""
    input_dir = os.path.dirname(input_file)
    if not input_dir:
        input_dir = '.'
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    avg_metrics_file = os.path.join(input_dir, f"{base_name}_avg_metrics.csv")
    detailed_results_file = os.path.join(input_dir, f"{base_name}_detailed_results.csv")
    
    try:
        avg_metrics_df.to_csv(avg_metrics_file, index=False, float_format='%.4f')
        detailed_results_df.to_csv(detailed_results_file, index=False, float_format='%.4f')
        
        print(f"\nAverage metrics saved to {avg_metrics_file}")
        print(f"Detailed results saved to {detailed_results_file}")
        
        print("\n=== Evaluation Summary ===")
        for _, row in avg_metrics_df.iterrows():
            print(f"{row['metric']}: {row['value']:.4f}")
            
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate code translation results using CodeBLEU.')
    parser.add_argument('--input', '-i', type=str,default="/path/to/test_trans.csv",
                        help='Input CSV file path with translation results. Columns: index, code, reference, generated.')
    parser.add_argument('--lang', '-l', type=str,default="java",
                        help='Target language of the generated and reference code (e.g., java, python).')
    parser.add_argument('--source_lang', '-sl', type=str, default="python", required=False,
                        help='Source language of the original code (e.g., python, java). Optional for some CodeBLEU setups.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found - {args.input}")
        sys.exit(1)
    
    is_summarization = "summary" in args.input.lower()
    
    if is_summarization:
        print("Detected 'summarization' task from input file name.")
        print("Using BLEU-based weights for CodeBLEU: (0.5, 0.5, 0.0, 0.0)")
        weights = (0.5, 0.5, 0.0, 0.0)

    else:
        print("Detected 'translation' task.")
        print("Using default CodeBLEU weights: (0.25, 0.25, 0.25, 0.25)")
        weights = (0.25, 0.25, 0.25, 0.25)
        
    avg_metrics_df, detailed_results_df = evaluate_translations(args.input, args.lang, args.source_lang, weights)
    
    if avg_metrics_df is not None and detailed_results_df is not None:
        save_evaluation_results(avg_metrics_df, detailed_results_df, args.input)
    else:
        print("Evaluation failed, no results generated.")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
python evaluate_translations.py --input /path/to/test_trans.csv

"""