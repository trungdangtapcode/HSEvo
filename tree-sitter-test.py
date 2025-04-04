from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# Load Python language
# PY_LANGUAGE = get_language('python')
PY_LANGUAGE = Language(tspython.language())

parser = Parser(PY_LANGUAGE)
# parser.set_language(PY_LANGUAGE)

def count_logical_nodes_and_complexity(code: str):
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    def traverse(node):
        count = 1  # count current node
        complexity = 0

        # Increase complexity for common decision points
        if node.type in {"if_statement", "for_statement", "while_statement",
                         "try_statement", "except_clause", "with_statement",
                         "match_statement", "case_clause", "conditional_expression",
                         "boolean_operator"}:
            complexity += 1

        for child in node.children:
            c, cx = traverse(child)
            count += c
            complexity += cx

        return count, complexity

    total_nodes, complexity = traverse(root_node)
    # Base complexity is 1
    return total_nodes, complexity + 1


def remove_comments_and_docstrings(code):
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    lines = code.split('\n')
    to_remove = []

    def is_comment(node):
        return node.type == 'comment'

    def is_docstring(node):
        return (node.type == 'string' and
                (node.parent.type in ['expression_statement', 'module']))

    def traverse(node):
        if is_comment(node):
            to_remove.append((node.start_point, node.end_point))
        elif is_docstring(node):
            to_remove.append((node.start_point, node.end_point))
        for child in node.children:
            traverse(child)

    traverse(root_node)

    for start, end in reversed(to_remove):
        start_row, start_col = start
        end_row, end_col = end
        if start_row == end_row:
            lines[start_row] = lines[start_row][:start_col] + lines[start_row][end_col:]
        else:
            lines[start_row] = lines[start_row][:start_col]
            for row in range(start_row + 1, end_row):
                lines[row] = ''
            lines[end_row] = lines[end_row][end_col:]

    cleaned_lines = [line for line in lines if line.strip() != '']

    return '\n'.join(cleaned_lines)

# Example usage
code = '''
import numpy as np
import random
import math
import scipy
import torch
def heuristics_v2(num_items: int, capacity: int, items: np.ndarray) -> np.ndarray:
    # items: ndarray of shape (num_items, 2), with columns [value, weight]
    
    # Compute value-to-weight ratio and store original indices
    indices = np.arange(num_items)
    ratios = items[:, 0] / items[:, 1]
    
    # Sort by ratio descending
    sorted_indices = np.argsort(-ratios)
    
    total_weight = 0
    res = np.zeros(num_items, dtype=int)
    
    for idx in sorted_indices:
        value, weight = items[idx]
        if total_weight + weight <= capacity:
            total_weight += weight
            res[idx] = 1
    
    return res


'''

code = remove_comments_and_docstrings(code)

nodes, complexity = count_logical_nodes_and_complexity(code)
print(f"Logical Nodes: {nodes}")
print(f"Cyclomatic Complexity: {complexity}")
