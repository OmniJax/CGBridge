import argparse
import pandas as pd
from openai import OpenAI, APIError
import concurrent.futures
import time
import os
from tqdm import tqdm
import itertools
import threading
import json

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge: Use LLM API to evaluate code summary quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required parameters
    parser.add_argument(
        "input_csv", 
        type=str,
        help="Input CSV file path, containing idx, code, reference, generated fields"
    )
    
    # Optional parameters
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV file path. If not specified, '-llm_as_judge.csv' will be appended to the input filename"
    )
    
    parser.add_argument(
        "--max-threads", "-t",
        type=int,
        default=5,
        help="Maximum number of concurrent threads"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="API request timeout (seconds)"
    )
    
    parser.add_argument(
        "--retry-attempts", "-r",
        type=int,
        default=3,
        help="Number of retry attempts for API request failures"
    )
    parser.add_argument(
        "--config_path", "-c",
        type=str,
        default=None,
        help="API configuration file path"
    )
    
    parser.add_argument(
        "--retry-delay", "-d",
        type=int,
        default=5,
        help="Delay (seconds) for retrying API request failures"
    )
    
    parser.add_argument(
        "--evaluation-column",
        type=str,
        default="evaluation_result",
        help="Column name to store evaluation JSON results"
    )
    
    parser.add_argument(
        "--nrows", "-n",
        type=int,
        default=None,
        help="Process only the first N rows of data (for testing or limiting processing volume)"
    )
    
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Set file path configuration
INPUT_CSV_PATH = args.input_csv
# Modify output file path, adding row count identifier to filename when using nrows
if args.output:
    OUTPUT_CSV_PATH = args.output
else:
    base_name = args.input_csv.rsplit('.', 1)[0]
    extension = args.input_csv.rsplit('.', 1)[1] if '.' in args.input_csv else 'csv'
    
    if args.nrows is not None:
        OUTPUT_CSV_PATH = f"{base_name}-llm_as_judge-n{args.nrows}.{extension}"
    else:
        OUTPUT_CSV_PATH = f"{base_name}-llm_as_judge.{extension}"

EVALUATION_COLUMN = args.evaluation_column
MAX_THREADS = args.max_threads
REQUEST_TIMEOUT = args.timeout
RETRY_ATTEMPTS = args.retry_attempts
RETRY_DELAY = args.retry_delay
NROWS = args.nrows  

print("=" * 60)
print("LLM-as-Judge Code Summary Evaluation Tool")
print("=" * 60)
print(f"Input file: {INPUT_CSV_PATH}")
print(f"Output file: {OUTPUT_CSV_PATH}")
print(f"Maximum threads: {MAX_THREADS}")
print(f"Request timeout: {REQUEST_TIMEOUT} seconds")
print(f"Retry attempts: {RETRY_ATTEMPTS}")
print(f"Retry delay: {RETRY_DELAY} seconds")
print(f"Evaluation result column name: {EVALUATION_COLUMN}")
if NROWS is not None:
    print(f"Row limit for processing: First {NROWS} rows")
else:
    print(f"Row limit for processing: Unlimited (process all data)")
print("=" * 60)

# Verify if input file exists
if not os.path.exists(INPUT_CSV_PATH):
    print(f"Error: Input file {INPUT_CSV_PATH} does not exist")
    exit(1)

# --- Configuration ---
def load_api_config():
    """Load API configuration from JSON file"""
    import json
    import os
    
    
    config_paths = [args.config_path]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"Successfully loaded API configuration from {config_path}")
                return config
            except Exception as e:
                print(f"Failed to load configuration from {config_path}: {e}")
                continue
    
    raise FileNotFoundError("No valid api_keys.json configuration file found")

# Load configuration
try:
    config = load_api_config()
    API_KEYS = config.get("api_keys", [])
    API_URL = config.get("api_url", " ")
    MODEL_NAME = config.get("model_name", "gpt-4o-mini")
    
    if not API_KEYS or not API_URL or not MODEL_NAME:
        raise ValueError("API_KEYS, API_URL, MODEL_NAME cannot be empty")
    
    print(f"Loaded {len(API_KEYS)} API keys")
    print(f"API URL: {API_URL}")
    print(f"Model: {MODEL_NAME}")
    
except Exception as e:
    print(f"Configuration loading failed: {e}")
    raise e

# Global API Key cycler and lock - initialized after loading configuration
if not API_KEYS:
    raise ValueError("API_KEYS list cannot be empty.")
api_key_cycle = itertools.cycle(API_KEYS)
api_key_lock = threading.Lock()

print(f"API Key rotator initialized, supporting concurrent rotation of {len(API_KEYS)} keys")

# JSON Schema defining evaluation structure
EVALUATION_JSON_SCHEMA = {
    "name": "code_summary_evaluation",
    "schema": {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "Brief textual analysis explaining the summary's main strengths and weaknesses"
            },
            "ratings": {
                "type": "object",
                "properties": {
                    "coherence": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 4,
                        "description": "How logically organized and well-structured is the summary (0-4)"
                    },
                    "consistency": {
                        "type": "integer", 
                        "minimum": 0,
                        "maximum": 4,
                        "description": "Does the summary accurately reflect the code's functionality (0-4)"
                    },
                    "fluency": {
                        "type": "integer",
                        "minimum": 0, 
                        "maximum": 4,
                        "description": "Is the summary written in clear, natural language (0-4)"
                    },
                    "relevance": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 4, 
                        "description": "Does the summary capture essential information without redundancy (0-4)"
                    }
                },
                "required": ["coherence", "consistency", "fluency", "relevance"],
                "additionalProperties": False
            },
            "overall": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 4,
                        "description": "Holistic overall score (0-4)"
                    },
                    "justification": {
                        "type": "string",
                        "description": "Brief justification for the overall score"
                    }
                },
                "required": ["score", "justification"],
                "additionalProperties": False
            }
        },
        "required": ["analysis", "ratings", "overall"],
        "additionalProperties": False
    },
    "strict": True
}

# LLM-as-Judge prompt template
SYSTEM_PROMPT = """You are an expert Principal Software Engineer acting as a meticulous Code Reviewer. Your sole task is to provide a critical and objective evaluation of a candidate code summary based on the provided source code.

Your evaluation must follow these steps:
1. Carefully read the source code to fully understand its functionality, inputs, outputs, and key logic.
2. Critically analyze the candidate summary against the code.
3. Provide a structured evaluation based on the four dimensions below.

**Evaluation Dimensions:**

* **Coherence (0-4)**: How logically organized and well-structured is the summary? Does it form a coherent description of the code? (0=Incoherent, 4=Perfectly coherent).
* **Consistency (0-4)**: Does the summary accurately reflect the code's functionality and logic? Are there any factual errors or hallucinations? This is the most critical dimension. (0=Contradicts the code, 4=Perfectly consistent).
* **Fluency (0-4)**: Is the summary written in clear, natural, and grammatically correct language? (0=Unreadable, 4=Perfectly fluent).
* **Relevance (0-4)**: Does the summary capture the essential information without including redundant or trivial details? (0=Irrelevant, 4=Perfectly relevant).

**Overall Score:**

After rating the four dimensions, provide a holistic **Overall Score (0-4)**. This score is NOT a simple average. You must weigh **Consistency** most heavily, as an inconsistent summary is fundamentally flawed, regardless of its fluency."""


USER_PROMPT_TEMPLATE = """Please evaluate the following code summary.

### Source Code:
```python
{code}
```

### Candidate Summary:
{candidate}"""

# --- API call function ---
def evaluate_summary_via_api(code: str, generated_summary: str):

    if not isinstance(code, str) or not code.strip():
        return "Error: Source code is empty or not a string"
    
    if not isinstance(generated_summary, str) or not generated_summary.strip():
        return "Error: Generated summary is empty or not a string"

    current_api_key = None
    with api_key_lock:
        current_api_key = next(api_key_cycle)

    try:
        client = OpenAI(
            api_key=current_api_key,
            base_url=API_URL,
        )
    except Exception as e:
        print(f"OpenAI client initialization failed with key ending ...{current_api_key[-4:] if current_api_key else 'N/A'}: {e}")
        return f"Error: OpenAI client initialization failed. Details: {str(e)}"

    user_prompt = USER_PROMPT_TEMPLATE.format(code=code, candidate=generated_summary)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(RETRY_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                timeout=REQUEST_TIMEOUT,
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": EVALUATION_JSON_SCHEMA
                }
            )
            
            # Add type checking and debug information
            if isinstance(completion, str):
                error_msg = f"Error: API returned string instead of completion object. Response: {completion[:200]}..."
                print(f"TYPE ERROR with key ...{current_api_key[-4:]} for code (first 50 chars: '{code[:50]}...'): {error_msg}")
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    return error_msg
            
            # Check if completion object has expected property
            if not hasattr(completion, 'choices'):
                error_msg = f"Error: Completion object missing 'choices' attribute. Type: {type(completion)}, Attributes: {dir(completion)}"
                print(f"STRUCTURE ERROR with key ...{current_api_key[-4:]} for code (first 50 chars: '{code[:50]}...'): {error_msg}")
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    return error_msg
            
            if completion.choices and completion.choices[0].message:
                response_content = completion.choices[0].message.content
                if response_content:
                    # Since json_schema is used, OpenAI will ensure valid JSON
                    # But we still verify the structure is correct
                    try:
                        parsed_json = json.loads(response_content.strip())
                        # Verify required fields exist
                        required_fields = ['analysis', 'ratings', 'overall']
                        if all(field in parsed_json for field in required_fields):
                            return response_content.strip()
                        else:
                            missing_fields = [field for field in required_fields if field not in parsed_json]
                            error_details = f"Missing required fields: {missing_fields}"
                            final_error_msg = f"Error: JSON response missing required fields. Details: {error_details}"
                            print(f"JSON STRUCTURE ERROR with key ...{current_api_key[-4:]} for code (first 50 chars: '{code[:50]}...'): {final_error_msg}")
                            if attempt < RETRY_ATTEMPTS - 1:
                                time.sleep(RETRY_DELAY)
                                continue
                            else:
                                return final_error_msg
                    except json.JSONDecodeError as json_error:
                        error_details = f"Invalid JSON response: {json_error}"
                        final_error_msg = f"Error: API response is not valid JSON. Details: {error_details}"
                        print(f"JSON PARSE ERROR with key ...{current_api_key[-4:]} for code (first 50 chars: '{code[:50]}...'): {final_error_msg}")
                        if attempt < RETRY_ATTEMPTS - 1:
                            time.sleep(RETRY_DELAY)
                            continue
                        else:
                            return final_error_msg
                else:
                    error_details = "Response message content is empty or None"
                    final_error_msg = f"Error: API response did not contain evaluation result. Details: {error_details}"
                    print(f"API LOGIC ERROR (No Content) with key ...{current_api_key[-4:]} for code (first 50 chars: '{code[:50]}...'): {final_error_msg}")
                    if attempt < RETRY_ATTEMPTS - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        return final_error_msg
            else:
                final_error_msg = "Error: Unexpected API response structure (no choices or message)"
                print(f"API LOGIC ERROR (Bad Structure) with key ...{current_api_key[-4:]} for code (first 50 chars: '{code[:50]}...'): {final_error_msg}")
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    return final_error_msg

        except APIError as e:
            print(f"OpenAI API ERROR (Attempt {attempt + 1}/{RETRY_ATTEMPTS}) with key ...{current_api_key[-4:]} for code (first 50 chars: '{code[:50]}...'): {type(e).__name__} - {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                return f"Error: OpenAI API request failed after {RETRY_ATTEMPTS} attempts with key ...{current_api_key[-4:]}. Details: {type(e).__name__} - {str(e)}"
        except Exception as e:
            print(f"UNEXPECTED ERROR during OpenAI API processing (Attempt {attempt + 1}/{RETRY_ATTEMPTS}) with key ...{current_api_key[-4:]} for code (first 50 chars: '{code[:50]}...'): {type(e).__name__} - {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                return f"Error: Unexpected error with OpenAI API after {RETRY_ATTEMPTS} attempts with key ...{current_api_key[-4:]}. Details: {type(e).__name__} - {str(e)}"
    
    return f"Error: API request processing failed after exhausting all attempts with key ...{current_api_key[-4:]}."

# --- Main logic ---
def main():
    df_to_process = None
    df_target_for_saving = None
    is_reprocessing = False

    print(f"Attempting to load existing output file: {OUTPUT_CSV_PATH}")
    try:
        # Apply row limit when reading output file
        if NROWS is not None:
            df_existing_output = pd.read_csv(OUTPUT_CSV_PATH, nrows=NROWS)
            print(f"Successfully loaded: {OUTPUT_CSV_PATH} (limited to first {NROWS} rows)")
        else:
            df_existing_output = pd.read_csv(OUTPUT_CSV_PATH)
            print(f"Successfully loaded: {OUTPUT_CSV_PATH}")

        # Check and add evaluation-related columns
        evaluation_columns = [
            EVALUATION_COLUMN,  # Original JSON result
            'overall_score',
            'coherence',
            'consistency', 
            'fluency',
            'relevance',
            'ratings_average',  # New: average of four ratings
            'analysis',
            'justification'
        ]
        
        for col in evaluation_columns:
            if col not in df_existing_output.columns:
                df_existing_output[col] = None
            
        # Ensure columns are string type for .str.startswith
        df_existing_output[EVALUATION_COLUMN] = df_existing_output[EVALUATION_COLUMN].astype(str)

        # Find rows that need reprocessing (evaluation result is error or empty)
        needs_reprocessing = (
            df_existing_output[EVALUATION_COLUMN].str.startswith("Error", na=False) |
            df_existing_output[EVALUATION_COLUMN].isna() |
            (df_existing_output[EVALUATION_COLUMN] == 'None') |
            df_existing_output['overall_score'].isna()  # 如果评分列为空也重新处理
        )
        
        if needs_reprocessing.any():
            df_to_process = df_existing_output[needs_reprocessing].copy()
            df_target_for_saving = df_existing_output
            is_reprocessing = True
            print(f"Found {len(df_to_process)} rows to reprocess.")
        else:
            print(f"No rows to reprocess found in {OUTPUT_CSV_PATH}. Script will not execute any operations.")
            return

    except FileNotFoundError:
        print(f"Output file {OUTPUT_CSV_PATH} not found. Starting fresh processing from input file {INPUT_CSV_PATH}.")
        try:
            # Apply row limit when reading input file
            if NROWS is not None:
                df_to_process = pd.read_csv(INPUT_CSV_PATH, nrows=NROWS)
                print(f"Successfully read input file: {INPUT_CSV_PATH} (limited to first {NROWS} rows)")
            else:
                df_to_process = pd.read_csv(INPUT_CSV_PATH)
                print(f"Successfully read input file: {INPUT_CSV_PATH} (all data)")
            
            df_target_for_saving = df_to_process.copy()
            
            # Initialize evaluation columns for fresh processing
            evaluation_columns = [
                EVALUATION_COLUMN,  # Original JSON result
                'overall_score',
                'coherence', 
                'consistency',
                'fluency',
                'relevance',
                'ratings_average',  # New: average of four ratings
                'analysis',
                'justification'
            ]
            
            for col in evaluation_columns:
                df_target_for_saving[col] = pd.Series([None] * len(df_target_for_saving), dtype=object)
                
        except FileNotFoundError:
            print(f"Error: Input file {INPUT_CSV_PATH} not found. Please check the file path.")
            return
        except Exception as e:
            print(f"Error: Failed to read input file {INPUT_CSV_PATH}: {e}")
            return
    except Exception as e:
        print(f"Error: Failed to load or process existing output file {OUTPUT_CSV_PATH}: {e}. Please check the file or consider deleting it for fresh processing.")
        return

    if df_to_process is None or df_to_process.empty:
        print("No data to process.")
        return

    # Check if required columns exist
    required_columns = ['idx', 'code', 'reference', 'generated']
    missing_columns = [col for col in required_columns if col not in df_to_process.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return

    total_rows_to_process = len(df_to_process)
    total_input_rows = len(df_target_for_saving)  # 实际加载的总行数
    
    if NROWS is not None:
        print(f"A total of {total_rows_to_process} records will be submitted to the API for evaluation (filtered from the first {NROWS} rows).")
        print(f"Actual processing range: the first {total_input_rows} rows (limited by --nrows={NROWS})")
    else:
        print(f"A total of {total_rows_to_process} records will be submitted to the API for evaluation.")
        print(f"Actual processing range: all {total_input_rows} rows of data")

    future_to_task_info = {}

    def parse_and_store_result(original_idx, evaluation_result):
        """Parse evaluation result and store in respective columns"""
        try:
            if isinstance(evaluation_result, str) and not evaluation_result.startswith("Error"):
                # Parse JSON
                eval_data = json.loads(evaluation_result)
                
                # Store original JSON
                df_target_for_saving.loc[original_idx, EVALUATION_COLUMN] = evaluation_result
                
                # Extract ratings
                ratings = eval_data.get('ratings', {})
                coherence = ratings.get('coherence', None)
                consistency = ratings.get('consistency', None)
                fluency = ratings.get('fluency', None)
                relevance = ratings.get('relevance', None)
                
                # Store ratings
                df_target_for_saving.loc[original_idx, 'overall_score'] = eval_data.get('overall', {}).get('score', None)
                df_target_for_saving.loc[original_idx, 'coherence'] = coherence
                df_target_for_saving.loc[original_idx, 'consistency'] = consistency
                df_target_for_saving.loc[original_idx, 'fluency'] = fluency
                df_target_for_saving.loc[original_idx, 'relevance'] = relevance
                df_target_for_saving.loc[original_idx, 'analysis'] = eval_data.get('analysis', None)
                df_target_for_saving.loc[original_idx, 'justification'] = eval_data.get('overall', {}).get('justification', None)
                
                # Calculate average of four ratings
                rating_scores = [coherence, consistency, fluency, relevance]
                valid_ratings = [score for score in rating_scores if score is not None and isinstance(score, (int, float))]
                
                if len(valid_ratings) == 4:  # Ensure all four ratings are valid
                    ratings_average = sum(valid_ratings) / len(valid_ratings)
                    df_target_for_saving.loc[original_idx, 'ratings_average'] = round(ratings_average, 2)
                else:
                    df_target_for_saving.loc[original_idx, 'ratings_average'] = None
                
            else:
                # Store error information
                df_target_for_saving.loc[original_idx, EVALUATION_COLUMN] = evaluation_result
                # Other columns remain None/NaN
                
        except json.JSONDecodeError as e:
            error_msg = f"Error: JSON parsing failed - {e}"
            df_target_for_saving.loc[original_idx, EVALUATION_COLUMN] = error_msg
        except Exception as e:
            error_msg = f"Error: Result parsing exception - {e}"
            df_target_for_saving.loc[original_idx, EVALUATION_COLUMN] = error_msg

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for original_index, row in df_to_process.iterrows():
            code = row['code']
            generated_summary = row['generated']
            
            # Check if input data is valid
            if pd.notna(code) and pd.notna(generated_summary) and \
               isinstance(code, str) and isinstance(generated_summary, str) and \
               code.strip() and generated_summary.strip():
                
                future = executor.submit(evaluate_summary_via_api, code, generated_summary)
                future_to_task_info[future] = {'original_index': original_index}
            else:
                # If input data is invalid, mark as error
                error_msg = "Error: Invalid input data (code or generated summary is empty or not string)"
                parse_and_store_result(original_index, error_msg)
        
        total_tasks_submitted = len(future_to_task_info)
        if total_tasks_submitted == 0:
            print("No valid tasks submitted to API (possibly all input data is invalid).")
        else:
            print(f"Submitted {total_tasks_submitted} API evaluation tasks.")

            for future in tqdm(concurrent.futures.as_completed(future_to_task_info), 
                             total=total_tasks_submitted, desc="Evaluating code summaries"):
                task_info = future_to_task_info[future]
                retrieved_original_idx = task_info['original_index']
                
                try:
                    evaluation_result = future.result()
                    parse_and_store_result(retrieved_original_idx, evaluation_result)
                except Exception as exc:
                    error_message = f"Error: Exception during evaluation processing for index {retrieved_original_idx} - {exc}"
                    parse_and_store_result(retrieved_original_idx, error_message)

    # 保存结果
    try:
        df_target_for_saving.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        print(f"Processing completed. Results saved to: {OUTPUT_CSV_PATH}")
        
        # Statistics on evaluation results
        try:
            successful_evaluations = df_target_for_saving['overall_score'].notna().sum()
            error_evaluations = df_target_for_saving['overall_score'].isna().sum()
            total_evaluations = len(df_target_for_saving)
            success_rate = successful_evaluations / total_evaluations if total_evaluations > 0 else 0
            
            print(f"Evaluation statistics: {successful_evaluations} successful, {error_evaluations} errors")
            
            # Prepare summary statistics data
            summary_data = {
                'input_file': [INPUT_CSV_PATH],
                'output_file': [OUTPUT_CSV_PATH],
                'total_samples': [total_evaluations],
                'successful_evaluations': [successful_evaluations],
                'error_evaluations': [error_evaluations],
                'success_rate': [round(success_rate, 4)],
                'processing_timestamp': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
            }
            
            if successful_evaluations > 0:
                # Calculate average scores for each dimension (including new ratings_average)
                score_columns = ['overall_score', 'coherence', 'consistency', 'fluency', 'relevance', 'ratings_average']
                print(f"\nScore statistics:")
                
                for col in score_columns:
                    # Ensure data type is correct, filter out non-numeric data
                    try:
                        # Convert to numeric type, non-numeric will become NaN
                        numeric_scores = pd.to_numeric(df_target_for_saving[col], errors='coerce')
                        valid_scores = numeric_scores.dropna()
                        
                        if len(valid_scores) > 0:
                            avg_score = valid_scores.mean()
                            std_score = valid_scores.std()
                            min_score = valid_scores.min()
                            max_score = valid_scores.max()
                            
                            # 添加到汇总数据
                            summary_data[f'{col}_mean'] = [round(avg_score, 3)]
                            summary_data[f'{col}_std'] = [round(std_score, 3)]
                            summary_data[f'{col}_min'] = [min_score]
                            summary_data[f'{col}_max'] = [max_score]
                            if col == 'ratings_average':
                                print(f"  - {col} (Average of four dimensions): {avg_score:.2f}/4.0 (±{std_score:.2f})")
                            else:
                                print(f"  - {col}: {avg_score:.2f}/4.0 (±{std_score:.2f})")
                        else:
                            # If there are no valid numeric data
                            summary_data[f'{col}_mean'] = [None]
                            summary_data[f'{col}_std'] = [None]
                            summary_data[f'{col}_min'] = [None]
                            summary_data[f'{col}_max'] = [None]
                            print(f"  - {col}: No valid numeric data")
                            
                    except Exception as col_error:
                        print(f"  - {col}: Statistical calculation error - {col_error}")
                        summary_data[f'{col}_mean'] = [None]
                        summary_data[f'{col}_std'] = [None]
                        summary_data[f'{col}_min'] = [None]
                        summary_data[f'{col}_max'] = [None]
                
                # Overall score distribution
                try:
                    overall_scores_numeric = pd.to_numeric(df_target_for_saving['overall_score'], errors='coerce').dropna()
                    if len(overall_scores_numeric) > 0:
                        score_dist = overall_scores_numeric.value_counts().sort_index().to_dict()
                        print(f"\nOverall score distribution: {score_dist}")
                        
                        # Add distribution information to summary data
                        for score, count in score_dist.items():
                            # Ensure score is numeric
                            if isinstance(score, (int, float)) and not pd.isna(score):
                                summary_data[f'overall_score_{int(score)}'] = [count]
                except Exception as dist_error:
                    print(f"Overall score distribution calculation error: {dist_error}")
                
                # Average score distribution of four dimensions
                try:
                    ratings_avg_numeric = pd.to_numeric(df_target_for_saving['ratings_average'], errors='coerce').dropna()
                    if len(ratings_avg_numeric) > 0:
                        # Round average scores to the nearest 0.5 for grouping statistics
                        rounded_avg_scores = (ratings_avg_numeric * 2).round() / 2
                        avg_score_dist = rounded_avg_scores.value_counts().sort_index().to_dict()
                        print(f"Average score distribution of four dimensions: {avg_score_dist}")
                except Exception as avg_dist_error:
                    print(f"Average score distribution calculation error: {avg_dist_error}")
            
            else:
                # If there are no successful evaluations, fill with null values
                score_columns = ['overall_score', 'coherence', 'consistency', 'fluency', 'relevance', 'ratings_average']
                for col in score_columns:
                    summary_data[f'{col}_mean'] = [None]
                    summary_data[f'{col}_std'] = [None]
                    summary_data[f'{col}_min'] = [None]
                    summary_data[f'{col}_max'] = [None]
            
            # Create summary DataFrame and save - change to vertical arrangement
            summary_items = []
            
            # Basic information
            summary_items.extend([
                {'metric': 'input_file', 'value': INPUT_CSV_PATH},
                {'metric': 'output_file', 'value': OUTPUT_CSV_PATH},
                {'metric': 'processing_timestamp', 'value': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')},
                {'metric': 'nrows_limit', 'value': NROWS if NROWS is not None else 'unlimited'},  # New: record row limit
                {'metric': 'total_samples', 'value': total_evaluations},
                {'metric': 'successful_evaluations', 'value': successful_evaluations},
                {'metric': 'error_evaluations', 'value': error_evaluations},
                {'metric': 'success_rate', 'value': round(success_rate, 4)}
            ])
            
            if successful_evaluations > 0:
                # Statistics for each dimension's scores
                score_columns = ['overall_score', 'coherence', 'consistency', 'fluency', 'relevance', 'ratings_average']
                
                for col in score_columns:
                    try:
                        numeric_scores = pd.to_numeric(df_target_for_saving[col], errors='coerce')
                        valid_scores = numeric_scores.dropna()
                        
                        if len(valid_scores) > 0:
                            avg_score = valid_scores.mean()
                            std_score = valid_scores.std()
                            min_score = valid_scores.min()
                            max_score = valid_scores.max()
                            
                            summary_items.extend([
                                {'metric': f'{col}_mean', 'value': round(avg_score, 3)},
                                {'metric': f'{col}_std', 'value': round(std_score, 3)},
                                {'metric': f'{col}_min', 'value': min_score},
                                {'metric': f'{col}_max', 'value': max_score}
                            ])
                        else:
                            summary_items.extend([
                                {'metric': f'{col}_mean', 'value': None},
                                {'metric': f'{col}_std', 'value': None},
                                {'metric': f'{col}_min', 'value': None},
                                {'metric': f'{col}_max', 'value': None}
                            ])
                    except Exception as col_error:
                        print(f"  - {col}: Statistical calculation error - {col_error}")
                        summary_items.extend([
                            {'metric': f'{col}_mean', 'value': None},
                            {'metric': f'{col}_std', 'value': None},
                            {'metric': f'{col}_min', 'value': None},
                            {'metric': f'{col}_max', 'value': None}
                        ])
                
                # Score distribution statistics
                try:
                    overall_scores_numeric = pd.to_numeric(df_target_for_saving['overall_score'], errors='coerce').dropna()
                    if len(overall_scores_numeric) > 0:
                        score_dist = overall_scores_numeric.value_counts().sort_index().to_dict()
                        
                        # Add distribution information
                        for score, count in score_dist.items():
                            if isinstance(score, (int, float)) and not pd.isna(score):
                                summary_items.append({
                                    'metric': f'overall_score_{int(score)}_count', 
                                    'value': count
                                })
                except Exception as dist_error:
                    print(f"Overall score distribution calculation error: {dist_error}")
                
                # Average score distribution of four dimensions
                try:
                    ratings_avg_numeric = pd.to_numeric(df_target_for_saving['ratings_average'], errors='coerce').dropna()
                    if len(ratings_avg_numeric) > 0:
                        rounded_avg_scores = (ratings_avg_numeric * 2).round() / 2
                        avg_score_dist = rounded_avg_scores.value_counts().sort_index().to_dict()
                        
                        # Add average score distribution of four dimensions
                        for avg_score, count in avg_score_dist.items():
                            if isinstance(avg_score, (int, float)) and not pd.isna(avg_score):
                                summary_items.append({
                                    'metric': f'ratings_average_{avg_score:.1f}_count',
                                    'value': count
                                })
                except Exception as avg_dist_error:
                    print(f"Average score distribution calculation error: {avg_dist_error}")
            
            else:
                # If there are no successful evaluations, fill with null values
                score_columns = ['overall_score', 'coherence', 'consistency', 'fluency', 'relevance', 'ratings_average']
                for col in score_columns:
                    summary_items.extend([
                        {'metric': f'{col}_mean', 'value': None},
                        {'metric': f'{col}_std', 'value': None},
                        {'metric': f'{col}_min', 'value': None},
                        {'metric': f'{col}_max', 'value': None}
                    ])
            
            # Create vertical DataFrame
            summary_df = pd.DataFrame(summary_items)
            
            # Generate summary file path
            summary_csv_path = OUTPUT_CSV_PATH.rsplit('.', 1)[0] + '_summary.csv'
            
            # Check if summary file already exists, if so, append
            if os.path.exists(summary_csv_path):
                try:
                    existing_summary = pd.read_csv(summary_csv_path)
                    
                    # Add timestamp column to new data to distinguish different runs
                    current_timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    summary_df['run_id'] = current_timestamp
                    
                    # Rearrange column order: metric, value, run_id
                    summary_df = summary_df[['metric', 'value', 'run_id']]
                    
                    # If existing file does not have run_id column, add a default value
                    if 'run_id' not in existing_summary.columns:
                        existing_summary['run_id'] = 'previous_runs'
                        # Rearrange existing data's column order
                        existing_summary = existing_summary[['metric', 'value', 'run_id']]
                    
                    # Combine data
                    combined_summary = pd.concat([existing_summary, summary_df], ignore_index=True)
                    combined_summary.to_csv(summary_csv_path, index=False, encoding='utf-8')
                    print(f"Summary statistics appended to: {summary_csv_path}")
                    
                except Exception as e:
                    # If reading fails, overwrite directly
                    current_timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    summary_df['run_id'] = current_timestamp
                    summary_df = summary_df[['metric', 'value', 'run_id']]
                    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
                    print(f"Summary statistics saved to: {summary_csv_path} (overwrite mode)")
            else:
                # Create new file
                current_timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                summary_df['run_id'] = current_timestamp
                summary_df = summary_df[['metric', 'value', 'run_id']]
                summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
                print(f"Summary statistics saved to: {summary_csv_path}")
            
            # Display summary information
            print(f"\nSummary statistics:")
            print(f"  - Success rate: {success_rate:.1%}")
            if successful_evaluations > 0:
                # Find corresponding values from summary_items
                overall_mean = next((item['value'] for item in summary_items if item['metric'] == 'overall_score_mean'), None)
                ratings_avg_mean = next((item['value'] for item in summary_items if item['metric'] == 'ratings_average_mean'), None)
                
                if overall_mean is not None:
                    print(f"  - Overall average score: {overall_mean:.2f}/4.0")
                if ratings_avg_mean is not None:
                    print(f"  - Average score of four dimensions: {ratings_avg_mean:.2f}/4.0")
            
        except Exception as stats_error:
            print(f"统计结果时出错: {stats_error}")
            
    except Exception as e:
        print(f"错误: 保存结果到CSV文件 {OUTPUT_CSV_PATH} 失败: {e}")

def parse_evaluation_results(csv_path: str):
    """
    解析评价结果的辅助函数
    """
    try:
        df = pd.read_csv(csv_path)
        
        if EVALUATION_COLUMN not in df.columns:
            print(f"错误: CSV文件中未找到评价结果列 '{EVALUATION_COLUMN}'")
            return None
            
        evaluations = []
        for idx, row in df.iterrows():
            try:
                if isinstance(row[EVALUATION_COLUMN], str) and not row[EVALUATION_COLUMN].startswith("Error"):
                    eval_data = json.loads(row[EVALUATION_COLUMN])
                    eval_data['idx'] = row['idx']
                    evaluations.append(eval_data)
            except:
                continue
                
        return evaluations
        
    except Exception as e:
        print(f"解析评价结果时出错: {e}")
        return None

if __name__ == "__main__":
    import numpy as np  
    main()
    
    
    
    
"""

python llm-as-judge.py \
    /path/to/test_summary.csv \
    --max-threads 64 \
    --config_path /path/to/api_keys.json
    
    
"""
