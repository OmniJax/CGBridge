
import pandas as pd
import os
import re
import subprocess
import sys
import tempfile
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm


# def clean_java_syntax(code):
#     """Cleans common syntax issues in Java code"""
#     code = re.sub(r'\s*\.\s*', '.', code)
    
#     code = re.sub(r'\s*\(\s*', '(', code)
#     code = re.sub(r'\s*\)\s*', ')', code)
    
#     code = re.sub(r"'\s*([a-zA-Z0-9])\s*'", r"'\1'", code)
    
#     code = re.sub(r'\s*\[\s*\]', '[]', code)
    
#     code = re.sub(r'\s+', ' ', code)
    
#     return code

def extract_java_code_from_llm_output(raw_llm_output: str) -> str:
    """
    Extracts the first Java code block from the LLM's raw output.
    Prioritizes blocks marked with ```java ... ```, then generic ``` ... ```.
    If no blocks are found or extraction significantly alters plausible code,
    it attempts to remove common introductory/concluding phrases.
    If all cleaning attempts fail to yield a clearly better or still plausible code snippet,
    it returns the original input string (stripped).

    Args:
        raw_llm_output: The raw string output from the LLM.

    Returns:
        The extracted/cleaned Java code as a string, or the original stripped input
        if cleaning is unsuccessful or unnecessary.
    """
    if not isinstance(raw_llm_output, str):
        return "" # Or handle as an error, depending on desired behavior for non-string input

    original_stripped_output = raw_llm_output.strip()

    match_java = re.search(r"```java\s*([\s\S]*?)```", raw_llm_output, re.DOTALL)
    if match_java:
        return match_java.group(1).strip()

    match_generic = re.search(r"```([\s\S]*?)```", raw_llm_output, re.DOTALL)
    if match_generic:
        extracted_content = match_generic.group(1).strip()
        return extracted_content

    
    common_intros = [
        "Here's the Java translation of the given Python code:",
        "Here is the Java equivalent of your Python code:",
        "Here is the Java code:",
        "Certainly, here is the Java code:",
        "The Java translation is as follows:",
    ]
    common_outros = [ 
        "Explanation:",
        "This Java program does exactly the same thing",
        "This Java code performs the same operations",
        "Let me know if you have any other questions.",
    ]

    cleaned_output = original_stripped_output
    initial_cleaned_output_state = cleaned_output 
    removed_intro = False
    for intro in common_intros:
        if cleaned_output.startswith(intro):
            cleaned_output = cleaned_output[len(intro):].lstrip()
            removed_intro = True
            break
    
    initial_cleaned_output_for_outro_removal = cleaned_output 
    removed_outro = False
    
    last_code_char_index = -1
    idx_brace = cleaned_output.rfind('}')
    idx_semicolon = cleaned_output.rfind(';')
    
    if idx_brace != -1 or idx_semicolon != -1:
        last_code_char_index = max(idx_brace, idx_semicolon)

        if last_code_char_index != -1 and last_code_char_index < len(cleaned_output) -1 : 
            potential_code_part = cleaned_output[:last_code_char_index+1].rstrip()
            potential_explanation_part = cleaned_output[last_code_char_index+1:].lstrip()

            if potential_explanation_part: 
                for outro in common_outros:
                    if potential_explanation_part.startswith(outro) or outro in potential_explanation_part :
                        cleaned_output = potential_code_part 
                        removed_outro = True
                        break
                if not removed_outro and not ("import " in potential_explanation_part or "class " in potential_explanation_part or "public " in potential_explanation_part):
                    cleaned_output = potential_code_part
                    removed_outro = True 
    

    def is_plausible_code(text_to_check):
        if not text_to_check: # An empty string is not valid code
            return False
        has_java_keywords = "class " in text_to_check or \
                            "interface " in text_to_check or \
                            "enum " in text_to_check or \
                            "import " in text_to_check or \
                            "public " in text_to_check or \
                            "static " in text_to_check or \
                            "void " in text_to_check
        has_structure = text_to_check.count('{') > 0 and text_to_check.count('}') > 0 or \
                        text_to_check.count(';') > 0
        
        if len(text_to_check) < 20 and not (text_to_check.strip().startswith("import") or text_to_check.strip().startswith("public")):
            if not (has_java_keywords and has_structure): 
                 return False 

        if not has_java_keywords and not has_structure and len(text_to_check.split()) > 5: 
            is_only_explanation = True
            for intro in common_intros:
                if intro.strip().startswith(text_to_check.strip()): is_only_explanation=False; break
            for outro in common_outros:
                if outro.strip().startswith(text_to_check.strip()): is_only_explanation=False; break
            if is_only_explanation and not (text_to_check.count('{') > 0 or text_to_check.count(';') > 0) :
                return False


        return has_java_keywords or has_structure

    if (removed_intro or removed_outro) and is_plausible_code(cleaned_output):
        return cleaned_output.strip()
    
    if not (removed_intro or removed_outro) and is_plausible_code(original_stripped_output):
        return original_stripped_output

    if (removed_intro or removed_outro) and not is_plausible_code(cleaned_output) and is_plausible_code(original_stripped_output):
        print(f"Warning: Cleaning attempt resulted in less plausible code. Reverting to original for: \"{raw_llm_output[:100]}...\"")
        return original_stripped_output

    if not (match_java or match_generic):
        print(f"Info: No Markdown code blocks found. Returning original (stripped) input: \"{raw_llm_output[:100]}...\"")
        return original_stripped_output

    return original_stripped_output


# def add_common_imports(code):
#     common_imports = [
#         "import java.util.*;",
#         "import java.io.*;",
#         "import java.lang.*;",
#         "import java.math.*;",
#         "import java.text.*;",
#         "import java.time.*;"
#     ]

#     # Check if the code already has import statements
#     if "import " in code:
#         # Find the position of the first import statement
#         import_pos = code.find("import ")
#         # Insert common imports before the first import statement
#         return "\n".join(common_imports) + "\n" + code
#     else:
#         # If there are no import statements, add them before the class definition
#         if "class " in code:
#             class_pos = code.find("class ")
#             return "\n".join(common_imports) + "\n" + code
#         else:
#             # If there is no class definition, add them at the beginning
#             return "\n".join(common_imports) + "\n" + code
#     return "\n".join(common_imports) + "\n" + code



def extract_class_name(code):
    """Extracts the class name from Java code"""
    match = re.search(r'class\s+([A-Za-z0-9_]+)', code)
    if match:
        return match.group(1)
    return "Unknown" 

def compile_java_code(row_idx, code, temp_dir, code_type="code"):
    """Compiles Java code and returns the result"""
    try:
        task_temp_dir = os.path.join(temp_dir, f"{row_idx}_{code_type}")
        os.makedirs(task_temp_dir, exist_ok=True)
        
        code = extract_java_code_from_llm_output(code)

        class_name = extract_class_name(code)
        if class_name == "Unknown":
            class_name = "DefaultClass"
            if 'class ' in code:
                code = re.sub(r'class\s+[A-Za-z0-9_]+', f'class {class_name}', code)
            else:
                code = f"class {class_name} {{\n{code}\n}}"
        file_path = os.path.join(task_temp_dir, f"{class_name}.java")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        process = subprocess.run(
            ['javac', file_path], 
            capture_output=True, 
            text=True,
            timeout=10  # Set timeout to 10 seconds
        )
        
        if process.returncode == 0:
            return row_idx, True, "", class_name
        else:
            return row_idx, False, process.stderr, class_name
    except subprocess.TimeoutExpired:
        return row_idx, False, "Compilation timed out", "Unknown"
    except Exception as e:
        return row_idx, False, str(e), "Unknown"

def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Check the compilation status of Java code in a CSV file")
    parser.add_argument("--input", '-i', required=True, help="Path to the CSV file containing Java code")
    parser.add_argument("--threads", '-t', type=int, default=1, 
                       help="Number of threads to use, 1 for single-threaded (default), greater than 1 for multi-threaded")
    parser.add_argument("--keep-temp", action="store_true",
                       help="Keep the temporary directory, do not delete automatically")
    
    # New: Configurable field name parameters
    parser.add_argument("--code-fields", '-c', nargs='+', default=['tgt_code'],
                       help="List of code field names to check, can specify multiple, default is tgt_code")
    parser.add_argument("--index-field", '-idx', default='idx',
                       help="Field name to use as index, default is idx")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    csv_file = args.input
    thread_count = max(1, args.threads)  
    code_fields = args.code_fields  # List of code fields to check
    index_field = args.index_field  # Index field name
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        print(f"Successfully read CSV file, total {len(df)} records")
        
        # Check if necessary fields exist
        missing_fields = []
        if index_field not in df.columns:
            missing_fields.append(index_field)
        for field in code_fields:
            if field not in df.columns:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"Error: The following fields are missing in the CSV file: {missing_fields}")
            print(f"Available fields: {list(df.columns)}")
            sys.exit(1)
            
        print(f"Checking the following code fields: {code_fields}")
        print(f"Using index field: {index_field}")
        
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        sys.exit(1)
    
    # Create temporary directory
    csv_dir = os.path.dirname(os.path.abspath(csv_file))
    temp_dir_name = "java_compilation_temp"
    temp_dir = os.path.join(csv_dir, temp_dir_name)
    
    # If the directory exists, delete it first
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Create a new temporary directory
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Create a results dictionary to store compilation results for each code field
        all_results = {field: {} for field in code_fields}
        
        if thread_count == 1:
            print("Executing in single-threaded mode")
            # Single-threaded mode
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Compilation progress"):
                # Get the original index value
                original_index = row[index_field] if index_field in row else i
                
                # Process each code field
                for field in code_fields:
                    if field in row:
                        code = row[field]
                        if isinstance(code, str) and code.strip():
                            _, success, error, class_name = compile_java_code(i, code, temp_dir, field)
                            all_results[field][original_index] = (success, error, class_name)
                        else:
                            all_results[field][original_index] = (False, "Code is empty or not a string", "Unknown")
        else:
            print(f"Executing in multi-threaded mode, thread count: {thread_count}")
            # Multi-threaded mode
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                future_to_info = {}
                
                # Submit compilation tasks
                for i, row in tqdm(df.iterrows(), total=len(df), desc="Submitting tasks"):
                    original_index = row[index_field] if index_field in row else i
                    
                    # Process each code field
                    for field in code_fields:
                        if field in row:
                            code = row[field]
                            if isinstance(code, str) and code.strip():
                                future = executor.submit(compile_java_code, i, code, temp_dir, field)
                                future_to_info[future] = (original_index, field)
                            else:
                                all_results[field][original_index] = (False, "Code is empty or not a string", "Unknown")
                
                # Collect results
                for future in tqdm(as_completed(future_to_info), total=len(future_to_info), desc="Processing results"):
                    try:
                        _, success, error, class_name = future.result()
                        original_index, field = future_to_info[future]
                        all_results[field][original_index] = (success, error, class_name)
                    except Exception as e:
                        original_index, field = future_to_info[future]
                        error_msg = f"Error during processing: {str(e)}"
                        all_results[field][original_index] = (False, error_msg, "Unknown")
        
        # Add results to the original DataFrame
        # Create corresponding result columns for each code field
        for field in code_fields:
            df[f'{field}_compilation_success'] = df[index_field].apply(
                lambda idx: all_results[field].get(idx, (False, "", ""))[0]
            )
            df[f'{field}_error_message'] = df[index_field].apply(
                lambda idx: all_results[field].get(idx, (False, "", ""))[1]
            )
            df[f'{field}_class_name'] = df[index_field].apply(
                lambda idx: all_results[field].get(idx, (False, "", ""))[2]
            )
        
        # Save results
        output_file = os.path.splitext(csv_file)[0] + "_compilation_result.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        # Summarize results
        summary_data = {
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'total_records': [len(df)]
        }
        
        # Print summary results
        print("\nCompilation results summary:")
        print(f"Total records: {len(df)}")
        
        for field in code_fields:
            field_results = all_results[field]
            success_count = sum(1 for success, _, _ in field_results.values() if success)
            total_count = len(field_results)
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            print(f"\n{field} field:")
            print(f"  Total: {total_count}")
            print(f"  Success: {success_count} ({success_rate:.2f}%)")
            
            # Add to summary data
            summary_data[f'{field}_total'] = [total_count]
            summary_data[f'{field}_success'] = [success_count]
            summary_data[f'{field}_success_rate'] = [f"{success_rate:.2f}%"]
        
        # Save summary results
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.splitext(csv_file)[0] + "_compilation_metrics.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        print(f"\nDetailed results saved to: {output_file}")
        print(f"Summary results saved to: {summary_file}")
        
    finally:
        # Decide whether to keep the temporary directory based on parameters
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Cleaning up temporary directory: {temp_dir}")
        else:
            print(f"Keeping temporary directory: {temp_dir}")

if __name__ == "__main__":
    main()


"""

python check_java_compilation.py -t 128 -c generated --index-field idx -i  /path/to/test_trans.csv


""" 