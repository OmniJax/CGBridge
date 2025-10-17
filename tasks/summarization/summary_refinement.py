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

# --- Command Line Argument Configuration ---
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summary Refinement: Refine code summary quality using LLM APIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "input_csv", 
        type=str,
        help="Input CSV file path, must contain at least a 'code' column."
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV file path. If not specified, '-refined.csv' will be appended to the input filename."
    )
    
    parser.add_argument(
        "--max-threads", "-t",
        type=int,
        default=5,
        help="Maximum number of concurrent threads."
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="API request timeout in seconds."
    )
    
    parser.add_argument(
        "--retry-attempts", "-r",
        type=int,
        default=3,
        help="Number of retry attempts for failed API requests."
    )
    parser.add_argument(
        "--config_path", "-c",
        type=str,
        default=None,
        help="Path to the API configuration file."
    )
    
    parser.add_argument(
        "--retry-delay", "-d",
        type=int,
        default=5,
        help="Delay in seconds between retry attempts for failed API requests."
    )
    
    parser.add_argument(
        "--refinement-column",
        type=str,
        default="refined_summary",
        help="Column name to store the refined summaries."
    )
    
    parser.add_argument(
        "--nrows", "-n",
        type=int,
        default=None,
        help="Process only the first N rows of data (for testing or limiting processing)."
    )
    
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Configure file paths
INPUT_CSV_PATH = args.input_csv
REFINED_COLUMN = args.refinement_column

if args.output:
    OUTPUT_CSV_PATH = args.output
else:
    base_name = args.input_csv.rsplit('.', 1)[0]
    extension = args.input_csv.rsplit('.', 1)[1] if '.' in args.input_csv else 'csv'
    
    if args.nrows is not None:
        OUTPUT_CSV_PATH = f"{base_name}-refined-n{args.nrows}.{extension}"
    else:
        OUTPUT_CSV_PATH = f"{base_name}-refined.{extension}"

MAX_THREADS = args.max_threads
REQUEST_TIMEOUT = args.timeout
RETRY_ATTEMPTS = args.retry_attempts
RETRY_DELAY = args.retry_delay
NROWS = args.nrows

# Display configuration information
print("=" * 60)
print("🚀 Code Summary Refinement Tool")
print("=" * 60)
print(f"📁 Input File: {INPUT_CSV_PATH}")
print(f"📁 Output File: {OUTPUT_CSV_PATH}")
print(f"🔄 Max Threads: {MAX_THREADS}")
print(f"⏱️  Request Timeout: {REQUEST_TIMEOUT}s")
print(f"🔁 Retry Attempts: {RETRY_ATTEMPTS}")
print(f"⏳ Retry Delay: {RETRY_DELAY}s")
print(f"📋 Refinement Column: {REFINED_COLUMN}")
if NROWS is not None:
    print(f"📊 Row Limit: Processing first {NROWS} rows")
else:
    print(f"📊 Row Limit: No limit (processing all data)")
print("=" * 60)

# Validate input file existence
if not os.path.exists(INPUT_CSV_PATH):
    print(f"❌ Error: Input file {INPUT_CSV_PATH} not found.")
    exit(1)

# --- API Configuration ---
def load_api_config():
    """Load API configuration from a JSON file."""
    config_paths = [args.config_path] if args.config_path else []
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"✅ Successfully loaded API configuration from {config_path}")
                return config
            except Exception as e:
                print(f"❌ Failed to load configuration from {config_path}: {e}")
                continue
    
    raise FileNotFoundError("No valid API configuration file found.")

# Load configuration
try:
    config = load_api_config()
    API_KEYS = config.get("api_keys", [])
    API_URL = config.get("api_url", " ")
    MODEL_NAME = config.get("model_name", "gpt-4")
    
    if not API_KEYS or not API_URL or not MODEL_NAME:
        raise ValueError("API_KEYS, API_URL, and MODEL_NAME must not be empty.")
    
    print(f"📋 Loaded {len(API_KEYS)} API keys.")
    print(f"🌐 API URL: {API_URL}")
    print(f"🤖 Model: {MODEL_NAME}")
    
except Exception as e:
    print(f"❌ Configuration loading failed: {e}")
    raise e

# Global API Key cycler and lock
if not API_KEYS:
    raise ValueError("API_KEYS list cannot be empty.")
api_key_cycle = itertools.cycle(API_KEYS)
api_key_lock = threading.Lock()

print(f"🔄 API Key rotator initialized, supporting concurrent rotation of {len(API_KEYS)} keys.")

# --- Prompt Definition ---
SYSTEM_PROMPT = "You are an expert Python assistant. Generate clear, concise, and accurate docstrings strictly following PEP 257 triple quote formatting."

USER_PROMPT_TEMPLATE = """Generate a Python docstring for the code below.
Format: PEP 257 triple quotes.
Required Structure:
- One-line summary of the function. (Optional) More detailed explanations are provided if the logic is complex.
- Parameters: param_name (param_type): Description of parameter. (Use 'None' if no parameters)
- Returns: return_type: Description of returned value. (Use 'None' if no explicit return)
Code:
```python
{code}
```"""


# --- API Call Function ---
def refine_summary_via_api(code: str):
    """
    Refine a code summary using the LLM API.

    Args:
        code (str): The source code.

    Returns:
        str or None: The refined summary string on success, or an error message on failure.
    """
    if not isinstance(code, str) or not code.strip():
        return "Error: Source code is empty or not a string"

    current_api_key = None
    with api_key_lock:
        current_api_key = next(api_key_cycle)

    try:
        client = OpenAI(api_key=current_api_key, base_url=API_URL)
    except Exception as e:
        return f"Error: OpenAI client initialization failed. Details: {str(e)}"

    user_prompt = USER_PROMPT_TEMPLATE.format(code=code)
    
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
            )
            
            if completion.choices and completion.choices[0].message:
                response_content = completion.choices[0].message.content
                if response_content:
                    return response_content.strip()
            
            # If the response is empty or has an incorrect structure, retry.
            error_msg = "Error: API response is empty or has an unexpected structure"
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                return error_msg

        except APIError as e:
            print(f"API ERROR (Attempt {attempt + 1}/{RETRY_ATTEMPTS}) with key ...{current_api_key[-4:]}: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                return f"Error: API request failed after {RETRY_ATTEMPTS} attempts. Details: {e}"
        except Exception as e:
            print(f"UNEXPECTED ERROR (Attempt {attempt + 1}/{RETRY_ATTEMPTS}) with key ...{current_api_key[-4:]}: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                return f"Error: Unexpected error after {RETRY_ATTEMPTS} attempts. Details: {e}"
    
    return f"Error: Failed after all retry attempts."

# --- Main Logic ---
def main():
    df_to_process = None
    df_for_saving = None

    try:
        if os.path.exists(OUTPUT_CSV_PATH):
            print(f"Detected existing output file: {OUTPUT_CSV_PATH}. Performing incremental processing.")
            df_existing = pd.read_csv(OUTPUT_CSV_PATH, nrows=NROWS)
            df_for_saving = df_existing
            
            if REFINED_COLUMN not in df_for_saving.columns:
                df_for_saving[REFINED_COLUMN] = None

            # Ensure the column is of string type to use .str.startswith
            df_for_saving[REFINED_COLUMN] = df_for_saving[REFINED_COLUMN].astype(str)

            needs_reprocessing = (
                df_for_saving[REFINED_COLUMN].str.startswith("Error", na=True) |
                df_for_saving[REFINED_COLUMN].isna() |
                (df_for_saving[REFINED_COLUMN] == 'None') |
                (df_for_saving[REFINED_COLUMN].str.strip() == '')
            )
            
            if needs_reprocessing.any():
                df_to_process = df_for_saving[needs_reprocessing].copy()
                print(f"Found {len(df_to_process)} records to be (re)processed.")
            else:
                print("All records have been processed. No action needed.")
                return
        else:
            raise FileNotFoundError

    except FileNotFoundError:
        print(f"Output file {OUTPUT_CSV_PATH} not found. Starting fresh processing from the input file.")
        try:
            df_to_process = pd.read_csv(INPUT_CSV_PATH, nrows=NROWS)
            df_for_saving = df_to_process.copy()
            df_for_saving[REFINED_COLUMN] = None
        except FileNotFoundError:
            print(f"Error: Input file {INPUT_CSV_PATH} also not found.")
            return
        except Exception as e:
            print(f"Error: Failed to read input file {INPUT_CSV_PATH}: {e}")
            return
    
    if 'code' not in df_to_process.columns:
        print(f"Error: Input file must contain a 'code' column.")
        return

    total_rows_to_process = len(df_to_process)
    print(f"A total of {total_rows_to_process} records will be submitted to the API for refinement.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_index = {
            executor.submit(refine_summary_via_api, row['code']): index
            for index, row in df_to_process.iterrows()
            if pd.notna(row['code']) and isinstance(row['code'], str) and row['code'].strip()
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                         total=len(future_to_index), desc="Refining code summaries"):
            original_index = future_to_index[future]
            try:
                result = future.result()
                df_for_saving.loc[original_index, REFINED_COLUMN] = result
            except Exception as exc:
                error_msg = f"Error: Exception during processing: {exc}"
                df_for_saving.loc[original_index, REFINED_COLUMN] = error_msg

    # Save results
    try:
        df_for_saving.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        print(f"Processing complete. Results saved to: {OUTPUT_CSV_PATH}")

        # Tally results
        successful_count = (~df_for_saving[REFINED_COLUMN].str.startswith("Error", na=True)).sum()
        error_count = (df_for_saving[REFINED_COLUMN].str.startswith("Error", na=True)).sum()
        total_count = len(df_for_saving)
        success_rate = successful_count / total_count if total_count > 0 else 0

        print("\n--- Results Summary ---")
        print(f"Total rows processed: {total_count}")
        print(f"✅ Successful refinements: {successful_count}")
        print(f"❌ Failures: {error_count}")
        print(f"📊 Success Rate: {success_rate:.2%}")
        print("--------------------")

    except Exception as e:
        print(f"Error: Failed to save results to CSV file {OUTPUT_CSV_PATH}: {e}")

if __name__ == "__main__":
    main()
