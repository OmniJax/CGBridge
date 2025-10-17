import pandas as pd
# import requests # No longer needed for the API call itself
from openai import OpenAI, APIError # Changed import
import concurrent.futures
import time
import os
from tqdm import tqdm
import itertools 
import threading 


API_URL = "..."

MODEL_NAME = 'gpt-4o' # <--- 请确保这是您要使用的模型

# 全局API Key循环器和锁
if not API_KEYS:
    raise ValueError("API_KEYS list cannot be empty.")
api_key_cycle = itertools.cycle(API_KEYS)
api_key_lock = threading.Lock()

PROMPT_TEMPLATE = """任务：修正并格式化提供的{language}代码。

核心要求：
1.  **修正语法**：确保代码能正确编译和无语法错误地运行。
2.  **代码格式化**：遵循 {language} 的标准代码风格。
3.  **保持原始逻辑和行为**：【绝对不能】更改代码的原始逻辑、算法和功能行为。修改【仅限于】语法修正和代码格式化。

输出指示：
* 您的回复【必须，且只能是】处理完毕、格式化后的纯 {language} 代码文本。
* 【严格禁止】包含任何形式的解释性文字、段落说明、注释、或任何非代码字符。
* 【特别强调】：您的输出**绝对不能包含**任何Markdown标记，尤其是三个反引号（```）所形成的代码块标记（例如 ```{language} ... ``` 或 ``` ... ```）。输出内容必须是纯粹的、未经任何封装的原始代码。

请直接输出处理后的代码。

代码如下：
```{language}
{code}
```
"""

INPUT_CSV_PATH = "/path/to/CGBridge/tasks/translation/xlcost_translate_datset/code_dataset/python2java_all_raw.csv"
OUTPUT_CSV_PATH = "/path/to/CGBridge/tasks/translation/xlcost_translate_datset/code_dataset/python2java_all_formatted.csv"
CODE_COLUMN_NAME = 'tgt_code'  # CSV中包含待格式化代码的列名
JAVA_CODE_COLUMN_NAME = 'tgt_code'  # CSV中包含待格式化Java代码的列名
PYTHON_CODE_COLUMN_NAME = 'src_code' # CSV中包含待格式化Python代码的列名
JAVA_FORMATTED_COLUMN = 'formatted_tgt_code'
PYTHON_FORMATTED_COLUMN = 'formatted_src_code'
MAX_THREADS = 20  # 根据API限制和您的机器性能调整
REQUEST_TIMEOUT = 60  # API请求超时时间（秒）
RETRY_ATTEMPTS = 3 # API请求失败重试次数
RETRY_DELAY = 5 # API请求失败重试延迟（秒）

# --- API 调用函数 (使用 OpenAI SDK 和轮换密钥) ---
def format_code_via_api(code_to_format: str, language: str):
    """
    使用LLM API格式化给定的代码片段 (通过 OpenAI SDK)，轮换使用API密钥。

    Args:
        code_to_format (str): 需要格式化的代码。
        language (str): 代码的语言 ("Java" 或 "Python")。

    Returns:
        str or None: 成功则返回格式化后的代码，失败则返回None。
    """
    if not isinstance(code_to_format, str) or not code_to_format.strip():
        return "Error: Input code is empty or not a string"

    current_api_key = None
    with api_key_lock: # 加锁以线程安全地获取下一个key
        current_api_key = next(api_key_cycle)
    
    # print(f"Thread {threading.get_ident()} using API Key ending with: ...{current_api_key[-4:]}") # 用于调试

    try:
        client = OpenAI(
            api_key=current_api_key, # 使用轮换得到的密钥
            base_url=API_URL,
        )
    except Exception as e:
        print(f"OpenAI client initialization failed with key ending ...{current_api_key[-4:] if current_api_key else 'N/A'}: {e}")
        return f"Error: OpenAI client initialization failed. Details: {str(e)}"

    question = PROMPT_TEMPLATE.format(code=code_to_format, language=language)
    
    messages = [{"role": 'user', "content": question}]

    for attempt in range(RETRY_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                timeout=REQUEST_TIMEOUT,
                temperature=0.0
            )
            
            if completion.choices and completion.choices[0].message:
                formatted_code = completion.choices[0].message.content
                if formatted_code:
                    return formatted_code.strip()
                else:
                    error_details = "Response message content is empty or None"
                    final_error_msg = f"Error: API response did not contain formatted code. Details: {error_details}"
                    print(f"API LOGIC ERROR (No Content) with key ...{current_api_key[-4:]} for code (first 50 chars: '{code_to_format[:50]}...'): {final_error_msg}. Full Completion: {completion}")
                    return final_error_msg
            else:
                final_error_msg = "Error: Unexpected API response structure (no choices or message)"
                print(f"API LOGIC ERROR (Bad Structure) with key ...{current_api_key[-4:]} for code (first 50 chars: '{code_to_format[:50]}...'): {final_error_msg}. Full Completion: {completion}")
                return final_error_msg

        except APIError as e:
            print(f"OpenAI API ERROR (Attempt {attempt + 1}/{RETRY_ATTEMPTS}) with key ...{current_api_key[-4:]} for code (first 50 chars: '{code_to_format[:50]}...'): {type(e).__name__} - {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                return f"Error: OpenAI API request failed after {RETRY_ATTEMPTS} attempts with key ...{current_api_key[-4:]}. Details: {type(e).__name__} - {str(e)}"
        except Exception as e:
            print(f"UNEXPECTED ERROR during OpenAI API processing (Attempt {attempt + 1}/{RETRY_ATTEMPTS}) with key ...{current_api_key[-4:]} for code (first 50 chars: '{code_to_format[:50]}...'): {type(e).__name__} - {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                return f"Error: Unexpected error with OpenAI API after {RETRY_ATTEMPTS} attempts with key ...{current_api_key[-4:]}. Details: {type(e).__name__} - {str(e)}"
    
    return f"Error: API request processing failed after exhausting all attempts (or an unexpected state occurred) with key ...{current_api_key[-4:]}."


# --- 主逻辑 ---
def main():
    df_to_process = None
    df_target_for_saving = None
    is_reprocessing = False

    print(f"尝试加载已存在的输出文件: {OUTPUT_CSV_PATH}")
    try:
        df_existing_output = pd.read_csv(OUTPUT_CSV_PATH)
        print(f"成功加载: {OUTPUT_CSV_PATH}")

        if JAVA_FORMATTED_COLUMN not in df_existing_output.columns:
            df_existing_output[JAVA_FORMATTED_COLUMN] = None
        if PYTHON_FORMATTED_COLUMN not in df_existing_output.columns:
            df_existing_output[PYTHON_FORMATTED_COLUMN] = None
            
        # 确保列是字符串类型以便使用 .str.startswith
        df_existing_output[JAVA_FORMATTED_COLUMN] = df_existing_output[JAVA_FORMATTED_COLUMN].astype(str)
        df_existing_output[PYTHON_FORMATTED_COLUMN] = df_existing_output[PYTHON_FORMATTED_COLUMN].astype(str)

        java_needs_reprocessing = df_existing_output[JAVA_FORMATTED_COLUMN].str.startswith("Error", na=False)
        python_needs_reprocessing = df_existing_output[PYTHON_FORMATTED_COLUMN].str.startswith("Error", na=False)
        
        rows_to_reprocess_mask = java_needs_reprocessing | python_needs_reprocessing
        
        if rows_to_reprocess_mask.any():
            df_to_process = df_existing_output[rows_to_reprocess_mask].copy()
            df_target_for_saving = df_existing_output
            is_reprocessing = True
            print(f"找到 {len(df_to_process)} 行记录中的错误需要重新处理。")
        else:
            print(f"在 {OUTPUT_CSV_PATH} 中未找到以 'Error' 开头的记录进行重处理。脚本将不会执行任何操作。")
            return

    except FileNotFoundError:
        print(f"输出文件 {OUTPUT_CSV_PATH} 未找到。将从输入文件 {INPUT_CSV_PATH} 开始全新处理。")
        try:
            df_to_process = pd.read_csv(INPUT_CSV_PATH)
            df_target_for_saving = df_to_process # 稍后会添加新列
            # 为全新处理初始化格式化列
            df_target_for_saving[JAVA_FORMATTED_COLUMN] = pd.Series([None] * len(df_target_for_saving), dtype=object)
            df_target_for_saving[PYTHON_FORMATTED_COLUMN] = pd.Series([None] * len(df_target_for_saving), dtype=object)
        except FileNotFoundError:
            print(f"错误: 输入文件 {INPUT_CSV_PATH} 也未找到。请检查文件路径。")
            return
        except Exception as e:
            print(f"错误: 读取输入文件 {INPUT_CSV_PATH} 失败: {e}")
            return
    except Exception as e:
        print(f"错误: 加载或处理现有输出文件 {OUTPUT_CSV_PATH} 失败: {e}。请检查文件或考虑删除它以进行全新处理。")
        return

    if df_to_process is None or df_to_process.empty:
        # 这个情况理论上已经被前面的逻辑覆盖，但作为安全检查
        print("没有数据需要处理。")
        return

    # 检查必要的原始代码列是否存在于待处理的DataFrame中
    if JAVA_CODE_COLUMN_NAME not in df_to_process.columns:
        print(f"错误: Java代码列 '{JAVA_CODE_COLUMN_NAME}' 未在待处理数据中找到。")
        return
    if PYTHON_CODE_COLUMN_NAME not in df_to_process.columns:
        print(f"错误: Python代码列 '{PYTHON_CODE_COLUMN_NAME}' 未在待处理数据中找到。")
        return

    total_rows_to_process = len(df_to_process)
    # total_rows = len(df_target_for_saving) # 这是整个目标DataFrame的行数
    print(f"共计 {total_rows_to_process} 条代码记录（Java或Python）将被提交给API进行处理/重处理。")

    future_to_task_info = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # 迭代 df_to_process 来提交任务
        # original_index 是 df_to_process 中的索引，它对应于 df_target_for_saving 中的原始索引
        for original_index, row in df_to_process.iterrows():
            # 只有当该特定代码类型需要重处理时才提交任务
            # 或者，如果不是重处理模式，则总是处理
            
            process_java = False
            if is_reprocessing:
                # 如果该行的Java代码之前是错误，现在就需要处理
                if df_target_for_saving.loc[original_index, JAVA_FORMATTED_COLUMN].startswith("Error"):
                    process_java = True
            else: # 全新处理模式，总是尝试处理
                process_java = True

            if process_java:
                java_code = row[JAVA_CODE_COLUMN_NAME]
                if pd.notna(java_code) and isinstance(java_code, str) and java_code.strip():
                    future_java = executor.submit(format_code_via_api, java_code, "Java")
                    future_to_task_info[future_java] = {'original_index': original_index, 'type': 'java'}
                elif is_reprocessing: # 如果是重处理且原始代码无效，保留之前的错误或标记为新错误
                    pass # 保留 df_target_for_saving 中已有的错误信息
                else: # 全新处理且原始代码无效
                    df_target_for_saving.loc[original_index, JAVA_FORMATTED_COLUMN] = "Error"
            
            process_python = False
            if is_reprocessing:
                if df_target_for_saving.loc[original_index, PYTHON_FORMATTED_COLUMN].startswith("Error"):
                    process_python = True
            else:
                process_python = True

            if process_python:
                python_code = row[PYTHON_CODE_COLUMN_NAME]
                if pd.notna(python_code) and isinstance(python_code, str) and python_code.strip():
                    future_python = executor.submit(format_code_via_api, python_code, "Python")
                    future_to_task_info[future_python] = {'original_index': original_index, 'type': 'python'}
                elif is_reprocessing:
                    pass
                else:
                    df_target_for_saving.loc[original_index, PYTHON_FORMATTED_COLUMN] = "Error"
        
        total_tasks_submitted = len(future_to_task_info)
        if total_tasks_submitted == 0:
            print("没有有效的任务被提交给API (可能所有待重处理的行原始代码都无效)。")
        else:
            print(f"共提交 {total_tasks_submitted} 个API格式化任务。")

            for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=total_tasks_submitted, desc="格式化代码中"):
                task_info = future_to_task_info[future]
                retrieved_original_idx = task_info['original_index'] # 这是df_target_for_saving中的索引
                code_type = task_info['type']
                
                try:
                    formatted_code_result = future.result()
                    if code_type == 'java':
                        df_target_for_saving.loc[retrieved_original_idx, JAVA_FORMATTED_COLUMN] = formatted_code_result
                    elif code_type == 'python':
                        df_target_for_saving.loc[retrieved_original_idx, PYTHON_FORMATTED_COLUMN] = formatted_code_result
                except Exception as exc:
                    error_message = f"Error: Exception during {code_type} code processing for index {retrieved_original_idx} - {exc}"
                    if code_type == 'java':
                        df_target_for_saving.loc[retrieved_original_idx, JAVA_FORMATTED_COLUMN] = error_message
                    elif code_type == 'python':
                        df_target_for_saving.loc[retrieved_original_idx, PYTHON_FORMATTED_COLUMN] = error_message

    # 保存结果
    try:
        df_target_for_saving.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        print(f"处理完成。结果已保存到: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"错误: 保存结果到CSV文件 {OUTPUT_CSV_PATH} 失败: {e}")

if __name__ == "__main__":
    main() 