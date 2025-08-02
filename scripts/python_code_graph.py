import tree_sitter_python as tspython
from tree_sitter import Language, Parser

import os
import sys
import json
import numpy as np
import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer,AutoModel,RobertaModel,RobertaTokenizer
import tree_sitter
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import graph_datasets_dir, models_dir

def initialize_tree_sitter():

    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    return parser

def build_node_map(code: str, parser: Parser) -> Tuple[tree_sitter.Node, Dict]:
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    
    node_map = {}
    
    def traverse(node, node_id=0):
        if node.is_named:
            node_map[node_id] = {
                'id': node_id,
                'type': node.type,
                'start_point': node.start_point,
                'end_point': node.end_point,
                # 'start_byte': node.start_byte,
                # 'end_byte': node.end_byte,
                'text': extract_node_text(node, code.split('\n')),
                # 'raw_text': code[node.start_byte:node.end_byte],
                # 'flattened_text': extract_node_text(node, code.split('\n')).replace('\n', ' ').strip(),
                'children': []
            }
            
            children_ids = []
            next_id = node_id + 1
            
            for child in node.children:
                if child.is_named:
                    child_id = next_id
                    next_id, _ = traverse(child, next_id)
                    children_ids.append(child_id)
            
            node_map[node_id]['children'] = children_ids
            return next_id, node_id
        
        return node_id, None
    
    _, root_id = traverse(root_node)
    return root_node, node_map

def extract_node_text(node, code_lines):
    start_point = node.start_point  
    end_point = node.end_point      
    
    if start_point[0] == end_point[0]:
        return code_lines[start_point[0]][start_point[1]:end_point[1]]
    else:
        text = ""
        text += code_lines[start_point[0]][start_point[1]:] + "\n"
        for i in range(start_point[0]+1, end_point[0]):
            text += code_lines[i] + "\n"
        text += code_lines[end_point[0]][:end_point[1]]
        return text

'''
    pd.DataFrame.from_dict(node_map).T.type.unique()
    ['module' 'function_definition' 'identifier' 'parameters'
    'default_parameter' 'none' 'comment' 'block' 'try_statement'
    'if_statement' 'comparison_operator' 'expression_statement' 'assignment'
    'call' 'argument_list' 'for_statement' 'integer' 'binary_operator'
    'subscript' 'pattern_list' 'expression_list' 'while_statement'
    'parenthesized_expression' 'string' 'string_start' 'string_content'
    'string_end' 'elif_clause' 'else_clause' 'augmented_assignment'
    'break_statement' 'continue_statement' 'return_statement' 'except_clause'
    'finally_clause']
'''

import pandas as pd
def build_cfg(node_map: Dict) -> List[Tuple[int, int, str, str]]:
    """从AST构建简单的控制流图"""
    cfg_edges = []
    
    def find_parent(node_id):
        for pid, info in node_map.items():
            if node_id in info.get('children', []):
                return pid
        return None
    def find_next_statement(stmt_id):
        parent_id = find_parent(stmt_id)
        if not parent_id:
            return None
        
        parent_info = node_map[parent_id]
        if 'children' not in parent_info:
            return None
        
        children = parent_info['children']
        try:
            idx = children.index(stmt_id)
            if idx + 1 < len(children):
                return children[idx + 1]
        except ValueError:
            pass
        
        return find_next_statement(parent_id)
    
    def find_parent_loop(node_id):
        parent_id = find_parent(node_id)
        if not parent_id:
            return None
        
        parent_info = node_map[parent_id]
        if parent_info['type'] in ['for_statement', 'while_statement']:
            return parent_id
        
        return find_parent_loop(parent_id)

    function_definitions = {}
    for node_id, info in node_map.items():
        if info['type'] == 'function_definition':
            func_name_node_id = None
            for child_id in info.get('children', []):
                child_info = node_map.get(child_id)
                if child_info and child_info['type'] == 'identifier':
                    func_name_node_id = child_id
                    break  
            if func_name_node_id:
                func_name = node_map[func_name_node_id]['text']
                function_definitions[func_name] = node_id 
    

    blocks = {}
    for node_id, info in node_map.items():
        if info['type'] == 'block':
            blocks[node_id] = info
            for i in range(len(info['children']) - 1):
                curr_child = info['children'][i]
                next_child = info['children'][i + 1]
                cfg_edges.append((curr_child, next_child, 'CFG', 'sequential execution'))
            
            if info['children']:  
                last_stmt = info['children'][-1]
                parent_of_block = find_parent(node_id)
                if parent_of_block:
                    next_stmt = find_next_statement(parent_of_block)
                    if next_stmt and node_map[parent_of_block]['type'] not in ['for_statement', 'while_statement']:
                        cfg_edges.append((last_stmt, next_stmt, 'CFG', 'block exit'))
                        
    control_nodes = {}
    for node_id, info in node_map.items():
        if info['type'] in ['if_statement', 'for_statement', 'while_statement', 'try_statement']:
            control_nodes[node_id] = info
    
    for node_id, info in control_nodes.items():
        if info['type'] == 'if_statement':
            for child_id in info['children']:
                child_info = node_map[child_id]
                if child_info['type'] == 'block':
                    cfg_edges.append((node_id, child_id, 'CFG', 'true branch'))
                elif child_info['type'] == 'else_clause':
                    cfg_edges.append((node_id, child_id, 'CFG', 'false branch'))
                elif child_info['type'] == 'elif_clause':
                    cfg_edges.append((node_id, child_id, 'CFG', 'alternate condition branch'))
                elif child_info['type'] in ['comparison_operator', 'binary_operator', 'identifier', 'parenthesized_expression']:
                    cfg_edges.append((node_id, child_id, 'CFG', 'condition evaluation'))
                    
            has_else = any(node_map[c]['type'] in ['else_clause', 'elif_clause'] for c in info['children'])
            if not has_else:
                next_stmt = find_next_statement(node_id)
                if next_stmt:
                    cfg_edges.append((node_id, next_stmt, 'CFG', 'condition false jump'))
        
        elif info['type'] in ['for_statement', 'while_statement']:
            loop_body = None
            for child_id in info['children']:
                child_info = node_map[child_id]
                if child_info['type'] == 'block':
                    loop_body = child_id
                    if info['type'] == 'for_statement':
                        cfg_edges.append((node_id, child_id, 'CFG', 'for loop body'))
                    else:
                        cfg_edges.append((node_id, child_id, 'CFG', 'while loop body'))
                elif child_info['type'] in ['identifier', 'call', 'binary_operator', 'comparison_operator', 'parenthesized_expression', 'integer']:
                    if info['type'] == 'for_statement':
                        cfg_edges.append((node_id, child_id, 'CFG', 'for loop iteration range'))
                    else:
                        cfg_edges.append((node_id, child_id, 'CFG', 'while loop condition'))
            
            next_stmt = find_next_statement(node_id)
            if next_stmt:
                cfg_edges.append((node_id, next_stmt, 'CFG', 'loop exit'))
            
            if loop_body and loop_body in blocks and blocks[loop_body]['children']:
                last_stmt = blocks[loop_body]['children'][-1]
                cfg_edges.append((last_stmt, node_id, 'CFG', 'loop back'))
        
        elif info['type'] == 'try_statement':   
            for child_id in info['children']:
                child_info = node_map[child_id]
                if child_info['type'] == 'block':
                    cfg_edges.append((node_id, child_id, 'CFG', 'try block'))
                elif child_info['type'] == 'except_clause':
                    cfg_edges.append((node_id, child_id, 'CFG', 'exception handler'))
                elif child_info['type'] == 'finally_clause':
                    cfg_edges.append((node_id, child_id, 'CFG', 'finally block'))
    
    
    for node_id, info in node_map.items():
        if info['type'] == 'break_statement':
            loop_id = find_parent_loop(node_id)
            if loop_id:
                next_stmt = find_next_statement(loop_id)
                if next_stmt:
                    cfg_edges.append((node_id, next_stmt, 'CFG', 'break jump'))

    for node_id, info in node_map.items():
        if info['type'] == 'call':
            call_node_id = node_id
            
            if not info.get('children'):
                continue

            callable_expr_id = info['children'][0] 
            callable_expr_info = node_map.get(callable_expr_id)

            if not callable_expr_info:
                continue

            func_name_str = None
            if callable_expr_info['type'] == 'identifier':
                func_name_str = callable_expr_info['text']
            elif callable_expr_info['type'] == 'attribute':
                attribute_children_ids = callable_expr_info.get('children', [])
                if attribute_children_ids:
                    last_child_id_of_attribute = attribute_children_ids[-1]
                    last_child_info = node_map.get(last_child_id_of_attribute)
                    if last_child_info and last_child_info['type'] == 'identifier':
                        func_name_str = last_child_info['text']
            
            if func_name_str and func_name_str in function_definitions:
                target_function_def_id = function_definitions[func_name_str]
                cfg_edges.append((call_node_id, target_function_def_id, 'CFG', 'function call'))
    return cfg_edges

   
def build_dfg(node_map: Dict) -> List[Tuple[int, int, str, str]]:
    """Implementing the GraphCodeBERT approach"""
    dfg_edges = []
    
    variable_states = {}
    
    def process_node(node_id, states):
        info = node_map[node_id]
        node_type = info['type']
        local_dfg = []
        
        if node_type == 'default_parameter':
            param_name = None
            param_value = None
            
            for child_id in info['children']:
                child_info = node_map[child_id]
                if child_info['type'] == 'identifier':
                    param_name = child_id
                else:
                    param_value = child_id
            
            if param_name is not None:
                name_info = node_map[param_name]
                var_name = name_info['text']
                
                if var_name not in states:
                    states[var_name] = []
                states[var_name].append((param_name, 'definition'))
                
                if param_value is not None:
                    sub_dfg, new_states = process_node(param_value, states.copy())
                    local_dfg.extend(sub_dfg)
                    
                    local_dfg.append((param_value, param_name, 'DFG', 'contributes to'))
        
        elif node_type == 'identifier':
            var_name = info['text']
            if var_name in states:
                for def_id, _ in states[var_name]:
                    if def_id != node_id:  
                        local_dfg.append((def_id, node_id, 'DFG', 'flows to'))
        
        elif node_type == 'assignment' or node_type == 'augmented_assignment':
            left_nodes = []
            right_nodes = []
            
            for child_id in info['children']:
                child_info = node_map[child_id]
                if not left_nodes and child_info['type'] in ['identifier', 'pattern_list', 'subscript']:
                    left_nodes.append(child_id)
                elif child_info['type'] not in ['=', '+=', '-=', '*=', '/=']:
                    right_nodes.append(child_id)
            
            for right_id in right_nodes:
                sub_dfg, states = process_node(right_id, states.copy())
                local_dfg.extend(sub_dfg)
            
            # 处理左侧节点，建立contributes to关系
            for left_id in left_nodes:
                left_info = node_map[left_id]
                
                if left_info['type'] == 'identifier':
                    # 简单变量赋值
                    var_name = left_info['text']
                    if var_name not in states:
                        states[var_name] = []
                    states[var_name].append((left_id, 'definition'))
                    
                    for right_id in right_nodes:
                        local_dfg.append((right_id,left_id, 'DFG', 'contributes to'))
                        
                elif left_info['type'] == 'pattern_list':
                    collect_identifiers(left_id, states, right_nodes, local_dfg)
                
                elif left_info['type'] == 'subscript':
                    handle_subscript(left_id, states, right_nodes, local_dfg)
        
        elif node_type == 'if_statement':
            condition_state = states.copy()
            then_state = states.copy()
            else_state = states.copy()
            
            for child_id in info['children']:
                child_info = node_map[child_id]
                if child_info['type'] in ['comparison_operator', 'binary_operator', 'parenthesized_expression']:
                    sub_dfg, condition_state = process_node(child_id, condition_state)
                    local_dfg.extend(sub_dfg)
            
            then_branches = []
            else_branches = []
            
            for child_id in info['children']:
                child_info = node_map[child_id]
                if child_info['type'] == 'block':
                    sub_dfg, then_state = process_node(child_id, then_state)
                    local_dfg.extend(sub_dfg)
                    then_branches.append(then_state)
                elif child_info['type'] in ['else_clause', 'elif_clause']:
                    sub_dfg, else_state = process_node(child_id, else_state)
                    local_dfg.extend(sub_dfg)
                    else_branches.append(else_state)
            
            states = merge_states([*then_branches, *else_branches])
        
        elif node_type in ['for_statement', 'while_statement']:
            for _ in range(2):
                loop_state = states.copy()
                
                for child_id in info['children']:
                    child_info = node_map[child_id]
                    if child_info['type'] in ['comparison_operator', 'binary_operator','call', 'identifier', 'parenthesized_expression']:
                        sub_dfg, loop_state = process_node(child_id, loop_state)
                        local_dfg.extend(sub_dfg)
                
                for child_id in info['children']:
                    child_info = node_map[child_id]
                    if child_info['type'] == 'block':
                        sub_dfg, loop_state = process_node(child_id, loop_state)
                        local_dfg.extend(sub_dfg)
                
                states = merge_states([states, loop_state])
        
        else:
            for child_id in info['children']:
                sub_dfg, new_states = process_node(child_id, states.copy())
                local_dfg.extend(sub_dfg)
                states = merge_states([states, new_states])
        
        return local_dfg, states
    
    def merge_states(states_list):
        merged = {}
        for state in states_list:
            for var, defs in state.items():
                if var not in merged:
                    merged[var] = []
                merged[var].extend(defs)
        
        for var in merged:
            merged[var] = list(set(merged[var]))
        
        return merged
    
    def collect_identifiers(pattern_id, states, right_nodes, dfg):
        pattern_info = node_map[pattern_id]
        for child_id in pattern_info['children']:
            child_info = node_map[child_id]
            if child_info['type'] == 'identifier':
                var_name = child_info['text']
                if var_name not in states:
                    states[var_name] = []
                states[var_name].append((child_id, 'definition'))
                
                for right_id in right_nodes:
                    dfg.append((right_id,child_id, 'DFG', 'contributes to'))
    
    def handle_subscript(subscript_id, states, right_nodes, dfg):
        subscript_info = node_map[subscript_id]
        for child_id in subscript_info['children']:
            child_info = node_map[child_id]
            if child_info['type'] == 'identifier':
                for right_id in right_nodes:
                    dfg.append((right_id,subscript_id, 'DFG', 'contributes to'))
                    
    
    def process_function_parameters(function_id, states, dfg):
        """处理函数定义中的参数"""
        function_info = node_map[function_id]
        
        for child_id in function_info['children']:
            child_info = node_map[child_id]
            if child_info['type'] == 'parameters':
                for param_id in child_info['children']:
                    param_info = node_map[param_id]
                    if param_info['type'] == 'default_parameter':
                        sub_dfg, states = process_node(param_id, states.copy())
                        dfg.extend(sub_dfg)
                    elif param_info['type'] == 'identifier':
                        var_name = param_info['text']
                        if var_name not in states:
                            states[var_name] = []
                        states[var_name].append((param_id, 'definition'))
    
    for root_id, info in node_map.items():
        if info['type'] == 'module':
            local_dfg, _ = process_node(root_id, {})
            dfg_edges.extend(local_dfg)
        elif info['type'] == 'function_definition':
            states = {}
            process_function_parameters(root_id, states, dfg_edges)
            local_dfg, _ = process_node(root_id, states)
            dfg_edges.extend(local_dfg)
    
    dfg_edges = list(set(dfg_edges))
    dfg_edges.sort(key=lambda x: x[0])
    
    return dfg_edges

# 迭代版本的节点编码
def encode_nodes_iterative(node_map: Dict, tokenizer, model) -> torch.Tensor:
    """使用迭代方式逐个编码节点"""
    device = model.device
    node_embeddings_list = []
    node_ids = sorted(node_map.keys())  # 确保顺序一致
    
    for node_id in node_ids:
        info = node_map[node_id]
        text = info['refined_text']
        if text:
            inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # 使用[CLS]token的表示作为整个代码片段的表示
                embedding = outputs.last_hidden_state[:, 0, :].cpu().squeeze()
                node_embeddings_list.append(embedding)
        else:
            # 对于没有文本的节点，使用零向量
            node_embeddings_list.append(torch.zeros(768))  # CodeBERT的维度是768
    
    # 将所有嵌入堆叠为一个大张量
    if node_embeddings_list:
        node_embeddings = torch.stack(node_embeddings_list)
    else:
        node_embeddings = torch.zeros((0, 768))
    
    return node_embeddings

# 批量版本的节点编码（带批处理大小）
def encode_nodes_batch(node_map: Dict, tokenizer, model, batch_size=256) -> torch.Tensor:
    """使用批量方式编码节点，可控制批处理大小"""
    device = model.device
    node_ids = sorted(node_map.keys())  # 确保顺序一致
    
    # 收集所有节点的文本
    node_texts = []
    for node_id in node_ids:
        text = node_map[node_id]['refined_text']
        if text:
            node_texts.append(text)
        else:
            node_texts.append('')  # 空文本用空字符串表示
    
    # 检查是否有节点文本
    if not node_texts:
        return torch.zeros((0, 768))
    
    # 分批处理节点文本
    all_embeddings = []
    for i in range(0, len(node_texts), batch_size):
        batch_texts = node_texts[i:i+batch_size]
        
        # 批量编码
        inputs = tokenizer(batch_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # 获取[CLS] token的嵌入
            all_embeddings.append(batch_embeddings)
    
    # del inputs,outputs
    # torch.cuda.empty_cache()
    
    # 合并所有批次的嵌入
    if all_embeddings:
        node_embeddings = torch.cat(all_embeddings, dim=0)
    else:
        node_embeddings = torch.zeros((0, 768))
    
    return node_embeddings

# 迭代版本的边编码
def encode_edges_iterative(all_edges: List, tokenizer, model) -> torch.Tensor:
    """使用迭代方式逐个编码边"""
    device = model.device
    
    # 检查是否有边
    if not all_edges:
        print('警告: 没有找到边')
        return torch.zeros((0, 768))
    
    edge_embeddings_list = []
    for src, dst, edge_label, edge_attr in all_edges:
        inputs = tokenizer(edge_attr, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用[CLS]token的表示作为整个边属性的表示
            embedding = outputs.last_hidden_state[:, 0, :].cpu().squeeze()
            edge_embeddings_list.append(embedding)
    
    # 将所有嵌入堆叠为一个大张量
    edge_embeddings = torch.stack(edge_embeddings_list)
    return edge_embeddings

# 批量版本的边编码（带批处理大小）
def encode_edges_batch(all_edges: List, tokenizer, model, batch_size=256) -> torch.Tensor:
    """使用批量方式编码边，可控制批处理大小"""
    device = model.device
    
    # 检查是否有边
    if not all_edges:
        print('警告: 没有找到边')
        return torch.zeros((0, 768))
    
    # 收集所有边的文本属性
    edge_texts = [edge_attr for _, _, _, edge_attr in all_edges]
    
    # 分批处理边文本
    all_embeddings = []
    for i in range(0, len(edge_texts), batch_size):
        batch_texts = edge_texts[i:i+batch_size]
        
        # 批量编码
        inputs = tokenizer(batch_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # 获取[CLS] token的嵌入
            all_embeddings.append(batch_embeddings)
    
    # 合并所有批次的嵌入
    if all_embeddings:
        edge_embeddings = torch.cat(all_embeddings, dim=0)
    else:
        edge_embeddings = torch.zeros((0, 768))
    
    return edge_embeddings

# 统一的编码函数（可设置批处理大小）
def encode_nodes(node_map: Dict, tokenizer, model, batch_size=256) -> torch.Tensor:
    """编码节点，可设置批处理大小"""
    if batch_size <= 0:  # 迭代处理每个节点
        return encode_nodes_iterative(node_map, tokenizer, model)
    elif batch_size >= len(node_map):  # 一次性处理所有节点
        return encode_nodes_batch(node_map, tokenizer, model, len(node_map))
    else:  # 分批处理节点
        return encode_nodes_batch(node_map, tokenizer, model, batch_size)

def encode_edges_with_cache(all_edges: List) -> torch.Tensor:
    """使用预计算的缓存编码边"""
    global EDGE_EMBEDDING_CACHE
    
    # 检查是否有边
    if not all_edges:
        print('警告: 没有找到边')
        return torch.zeros((0, 768))
    
    # 从缓存中获取嵌入
    edge_embeddings = []
    for _, _, _, edge_attr in all_edges:
        if edge_attr in EDGE_EMBEDDING_CACHE:
            edge_embeddings.append(EDGE_EMBEDDING_CACHE[edge_attr])
        else:
            print(f"警告: 未找到边类型 '{edge_attr}' 的预计算嵌入")
            edge_embeddings.append(torch.zeros(768))
    
    # 将嵌入堆叠为一个大张量
    edge_embeddings = torch.stack(edge_embeddings)
    return edge_embeddings



def encode_edges(all_edges: List, tokenizer, model, batch_size=256,edge_cache_path=None) -> torch.Tensor:
    """编码边，可设置批处理大小"""

        
    if not all_edges:
        print('警告: 没有找到边')
        return torch.zeros((0, 768))
        
    if edge_cache_path:
        load_edge_embedding_cache(edge_cache_path)
        return encode_edges_with_cache(all_edges)
    
    if batch_size <= 0:  # 迭代处理每条边
        return encode_edges_iterative(all_edges, tokenizer, model)
    elif batch_size >= len(all_edges):  # 一次性处理所有边
        return encode_edges_batch(all_edges, tokenizer, model, len(all_edges))
    else:  # 分批处理边
        return encode_edges_batch(all_edges, tokenizer, model, batch_size)




def determine_ast_relation(parent_type, child_type):
    """确定AST边的细分类型"""
    if parent_type == 'function_definition':
        if child_type == 'identifier':
            return 'has name'
        elif child_type == 'parameters':
            return 'has parameters'
        elif child_type == 'block':
            return 'has body'
    
    elif parent_type == 'if_statement':
        if child_type in ['comparison_operator', 'binary_operator']:
            return 'has condition'
        elif child_type == 'block':
            return 'has then body'
        elif child_type == 'else_clause':
            return 'has else body'
        elif child_type == 'elif_clause':
            return 'has elif branch'
    
    elif parent_type in ['for_statement', 'while_statement']:
        if child_type in ['comparison_operator', 'call', 'identifier']:
            return 'has condition'
        elif child_type == 'block':
            return 'has body'
    
    elif parent_type == 'assignment':
        if child_type == 'identifier' and is_left_side(parent_type, child_type):
            return 'has target'
        else:
            return 'has value'
    
    # 默认关系
    return 'contains'

def is_left_side(parent_type: str, child_type: str, child_index: int = 0) -> bool:
    """
    判断给定类型的子节点在父节点中是否处于"左侧"（作为赋值目标）
    
    Args:
        parent_type: 父节点的类型
        child_type: 子节点的类型
        child_index: 子节点在同类型兄弟节点中的索引（用于区分多个相同类型的子节点）
        
    Returns:
        bool: 如果子节点是赋值目标，则返回True
    """
    # 赋值语句中的左侧标识符
    if parent_type == 'assignment' and child_type == 'identifier' and child_index == 0:
        return True
    
    # 增强赋值语句中的左侧标识符
    if parent_type == 'augmented_assignment' and child_type == 'identifier' and child_index == 0:
        return True
    
    # 多重赋值语句（元组赋值）中的模式列表中的标识符
    if parent_type == 'pattern_list' and child_type == 'identifier':
        return True
    
    # 函数定义中的函数名
    if parent_type == 'function_definition' and child_type == 'identifier' and child_index == 0:
        return True
    
    # 类定义中的类名
    if parent_type == 'class_definition' and child_type == 'identifier' and child_index == 0:
        return True
    
    # 参数定义中的参数名
    if parent_type == 'default_parameter' and child_type == 'identifier' and child_index == 0:
        return True
    
    # for循环中的迭代变量
    if parent_type == 'for_statement' and child_type == 'identifier' and child_index == 0:
        return True
    
    # with语句中的别名赋值
    if parent_type == 'with_item' and child_type == 'identifier' and child_index > 0:
        return True
    
    # except子句中的异常别名
    if parent_type == 'except_clause' and child_type == 'identifier' and child_index > 0:
        return True
    
    # 推导式中的迭代变量
    if parent_type in ['list_comprehension', 'dictionary_comprehension', 'set_comprehension'] and child_type == 'identifier' and child_index == 0:
        return True
    
    # 导入语句中的别名
    if parent_type == 'import_from_statement' and child_type == 'identifier' and child_index > 0:
        return True
    
    return False


def extract_ast_edges(node_map: Dict) -> List[Tuple[int, int, str]]:
    """从节点映射中提取AST边"""
    ast_edges = []
    for node_id, info in node_map.items():
        for child_id in info['children']:
            child_info = node_map[child_id]
            relation_type = determine_ast_relation(info['type'], child_info['type'])

            ast_edges.append((node_id, child_id, 'AST',relation_type))
            
    return ast_edges



def build_pyg_graph(node_map: Dict, all_edges: List, node_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, edge_type_to_id=None) -> Data:
    """构建PyTorch Geometric格式的图"""
    
    if edge_type_to_id is None:
        edge_type_to_id = {'AST': 0, 'CFG': 1, 'DFG': 2}
    
    node_ids = sorted(node_map.keys())
    
    # collect node text
    # x_text = []
    # for node_id in node_ids:
    #     x_text.append(node_map[node_id]['refined_text'])
    
    edge_index = []
    edge_type = []
    # edge_text = []
    
    for src, dst, edge_label, edge_attr_text in all_edges:
        src_idx = node_ids.index(src)
        dst_idx = node_ids.index(dst)
        edge_index.append([src_idx, dst_idx])
        if isinstance(edge_label, str):
            edge_label = edge_type_to_id.get(edge_label, edge_label)
        
        edge_type.append(edge_label)
        # edge_text.append(edge_attr_text)
    
    data = Data(
        x=node_embeddings,  
        # x_text=x_text,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_type=torch.tensor(edge_type),
        edge_attr=edge_embeddings,  
        # edge_text=edge_text,
    )
    return data


def refine_node_text(node_map: Dict) -> Dict:
    """
    为所有节点添加精简文本字段：statement类型只保留第一行，其他类型保持原样
    """
    statement_types = [
        'if_statement', 'for_statement', 'while_statement', 'try_statement',
        'elif_clause', 'else_clause', 'except_clause', 'finally_clause',
        'function_definition', 'class_definition'
    ]
    
    for node_id, info in node_map.items():
        original_text = info['text']
        
        if info['type'] in statement_types:
            if '\n' in original_text:
                refined_text = original_text.split('\n')[0].strip()
            else:
                refined_text = original_text
        else:
            refined_text = original_text
        
        info['refined_text'] = refined_text
    
    return node_map

def code_to_graph(code: str, tokenizer, model, edge_types=['AST','CFG','DFG'], code_from_file=False, batch_size=256, edge_cache_path=None, edge_type_to_id=None) -> Data:
    """处理单个Python文件，提取AST、CFG、DFG，编码节点，并返回PyG格式的图"""
    if code_from_file:
        with open(code, 'r', encoding='utf-8') as f:
            code = f.read()
    
    edge_types=[edge_type.upper() for edge_type in edge_types]

    parser = initialize_tree_sitter()
    
    _, node_map = build_node_map(code, parser)
    
    node_map = refine_node_text(node_map)
    node_map = dict(sorted(node_map.items()))
    
    all_edges = []
    if 'CFG' in edge_types:
        cfg_edges = build_cfg(node_map)
        all_edges.extend(cfg_edges)
    if 'DFG' in edge_types:
        dfg_edges = build_dfg(node_map)
        all_edges.extend(dfg_edges)
    if 'AST' in edge_types:
        ast_edges = extract_ast_edges(node_map)
        all_edges.extend(ast_edges)  
    node_embeddings = encode_nodes(node_map, tokenizer, model,batch_size=batch_size)
    
    edge_embeddings = encode_edges(all_edges, tokenizer, model,batch_size=batch_size,edge_cache_path=edge_cache_path)
    
    graph = build_pyg_graph(node_map, all_edges, node_embeddings, edge_embeddings, edge_type_to_id)
    
    return graph

def save_graph(graph: Data, output_path: str):
    """保存PyG图到文件"""
    torch.save(graph, output_path)
    print(f"图已保存到 {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Process Python code file to generate AST, CFG, DFG graph and encode with CodeBERT')
    parser.add_argument('--file_path', type=str, help='Path to the Python file')
    parser.add_argument('--output', type=str, help='Output path for the PyG graph')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Model name or path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use: cuda:0, cuda:1, cpu, etc.')
    parser.add_argument('--batch_size',type=int,default=256,help='Batch size for encoding nodes and edges')
    parser.add_argument('--edge_cache_path',type=str,default=None,help='Path to the cache file')
    return parser.parse_args()

def initialize_encoder(model_name_or_path: str, device: str):
    """初始化代码编码器"""
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
    model = RobertaModel.from_pretrained(model_name_or_path, device_map=device)
    return tokenizer, model

EDGE_EMBEDDING_CACHE = {}
def load_edge_embedding_cache(edge_cache_path):
    global EDGE_EMBEDDING_CACHE
    if not EDGE_EMBEDDING_CACHE:
        EDGE_EMBEDDING_CACHE = torch.load(edge_cache_path,weights_only=False)
        print(f"loaded {len(EDGE_EMBEDDING_CACHE)} edge embedding caches")


if __name__ == "__main__":
    args = parse_args()
    
    # 设置默认输出路径
    if args.output is None:
        output_dir = os.path.join(graph_datasets_dir(), "single_files")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, "code_graph.pt")
    
    # 初始化编码器
    tokenizer, model = initialize_encoder(args.model_name_or_path, args.device)
    
    print('batch size:',args.batch_size)
    
    graph = code_to_graph(args.file_path, tokenizer, model,edge_types=['AST','CFG','DFG'],code_from_file=True,batch_size=args.batch_size,edge_cache_path=args.edge_cache_path)
    save_graph(graph, args.output)
    
    print("Done")
    print(f"N of nodes: {graph.num_nodes}")
    print(f"N of edges: {graph.num_edges}")
    

'''
    python python_code_graph.py \
        --file_path /path/to/toy.py \
        --model_name_or_path microsoft/unixcoder-base \
        --output /path/to/my_graph.pt \
        --device cuda:0 \
        --edge_cache_path /path/to/unixcoder_edge_embeddings_cache.pt
'''