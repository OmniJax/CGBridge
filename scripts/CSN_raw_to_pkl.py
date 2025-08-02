import gzip
import json
import pandas as pd
import glob
import os
import sys
import argparse

import re
from io import StringIO
import tokenize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import datasets_dir

def remove_comments_and_docstrings(source, lang):
    try:
        if lang in ['python']:
            try:
                io_obj = StringIO(source)
                out = ""
                prev_toktype = tokenize.INDENT
                last_lineno = -1
                last_col = 0
                
                try:
                    for tok in tokenize.generate_tokens(io_obj.readline):
                        token_type = tok[0]
                        token_string = tok[1]
                        start_line, start_col = tok[2]
                        end_line, end_col = tok[3]
                        ltext = tok[4]
                        if start_line > last_lineno:
                            last_col = 0
                        if start_col > last_col:
                            out += (" " * (start_col - last_col))
                        # Remove comments:
                        if token_type == tokenize.COMMENT:
                            pass
                        # This series of conditionals removes docstrings:
                        elif token_type == tokenize.STRING:
                            if prev_toktype != tokenize.INDENT:
                                if prev_toktype != tokenize.NEWLINE:
                                    if start_col > 0:
                                        out += token_string
                        else:
                            out += token_string
                        prev_toktype = token_type
                        last_col = end_col
                        last_lineno = end_line
                except tokenize.TokenError:
                    try:
                        no_comments = re.sub(r'#.*$', '', source, flags=re.MULTILINE)
                        pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
                        return re.sub(pattern, '', no_comments)
                    except:
                        return source
                
                temp = []
                for x in out.split('\n'):
                    if x.strip() != "":
                        temp.append(x)
                return '\n'.join(temp)
                
            except Exception as e:
                print(f"Error processing Python code: {str(e)}")
                return source
                
        elif lang in ['ruby']:
            return source
        else:
            try:
                def replacer(match):
                    s = match.group(0)
                    if s.startswith('/'):
                        return " "
                    else:
                        return s
                pattern = re.compile(
                    r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                    re.DOTALL | re.MULTILINE
                )
                temp = []
                for x in re.sub(pattern, replacer, source).split('\n'):
                    if x.strip() != "":
                        temp.append(x)
                return '\n'.join(temp)
            except Exception as e:
                print(f"Error processing {lang} code: {str(e)}")
                return source
                
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return source
    


def read_jsonl_gz_files(directory, split_name, language):
    """
    Read all jsonl.gz files in the directory and merge them into a DataFrame
    """
    file_pattern = os.path.join(directory, f"{language}_{split_name}_*.jsonl.gz")
    all_files = glob.glob(file_pattern)
    
    print(f"Found {len(all_files)} {split_name} files")
    
    all_data = []
    for i, file_path in enumerate(all_files):
        print(f"Processing file {i+1}/{len(all_files)}: {os.path.basename(file_path)}")
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    all_data.append(data)
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    all_data=pd.DataFrame(all_data)
    all_data=all_data[['code','docstring','func_name','code_tokens','docstring_tokens','language','partition']]  
    print('--------------------------------remove comments and docstrings--------------------------------')
    all_data['cleaned_code']=all_data['code'].apply(lambda x: remove_comments_and_docstrings(x,language))

    return all_data

def process_language_data(source_base, target_base, language):
    """
    Process all datasets for the specified language
    """
    print(f"\nProcessing {language} language data...")
    
    # Process all datasets (train, test, valid)
    for split in ['train', 'test', 'valid']:
        print(f"\nProcessing {split} dataset...")
        
        # Source directory and target file path
        source_dir = os.path.join(source_base, language, 'final', 'jsonl', split)
        target_file = os.path.join(target_base, language, f'{language}_{split}_data.pkl')
        
        # Ensure target directory exists
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
        # Read data
        df = read_jsonl_gz_files(source_dir, split, language)
        
        # Display dataset information
        print(f"\n{split} dataset information:")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        
        # Save data
        print(f"Saving {split} dataset...")
        df.to_pickle(target_file)
        print(f"{split} dataset saved to: {target_file}")
        
        # Display memory usage
        print(f"{split} data set memory usage:", df.memory_usage().sum() / 1024**2, "MB")
        
        # Clean up memory
        del df

def main():
    # Convert raw CodeSearchNet dataset to pkl files
    
    # Set command line arguments
    parser = argparse.ArgumentParser(description='Process code search dataset')
    parser.add_argument('--source', type=str, required=True,
                      help='Source data directory path (e.g., /path/to/code_search_net/data)')
    parser.add_argument('--target', type=str, default=str(datasets_dir()),
                      help=f'Target data directory path (default: {datasets_dir()})')
    parser.add_argument('--language', type=str, default='python',
                      help='Programming language to process (python/java/javascript/go/ruby/php/all)')
    
    args = parser.parse_args()
    
    # Supported languages list
    available_languages = ['python', 'java', 'javascript', 'go', 'ruby', 'php']
    
    # Determine which language to process
    languages_to_process = (
        available_languages if args.language == 'all' 
        else [args.language.lower()]
    )
    
    # 验证语言选择
    for lang in languages_to_process:
        if lang not in available_languages:
            print(f"Error: unsupported language {lang}")
            print(f"Supported languages: {', '.join(available_languages)}")
            return
    
    # Process data for each language
    for language in languages_to_process:
        try:
            process_language_data(args.source, args.target, language)
            print(f"\n{language} language data processing completed!")
        except Exception as e:
            print(f"\nError processing {language} language data: {str(e)}")
    
    print("\nAll data processing completed!")
    


if __name__ == '__main__':
    # Convert raw CodeSearchNet dataset to pkl files
    main()