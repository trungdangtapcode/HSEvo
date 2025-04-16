# %%writefile utils/utils.py
from litellm import completion
import os
import logging
import concurrent.futures
import time
import re
import inspect
from dotenv import load_dotenv
from os.path import join, dirname

dotenv_path = join(dirname(__file__),'..', '.env')
load_dotenv(dotenv_path)

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()


def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found


def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if len(log) > 0:
            if log_status and "Traceback" in log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            else:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successful!")
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r'<start>(.*?)```python', r'<start>(.*?)<end>']
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


def multi_chat_completion(messages_list: list[list[dict]], n, model, temperature, isMultiLLM = False):
    """
    An example of messages_list:

    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {"role": "system", "content": "You are a knowledgeable guide."},
            {"role": "user", "content": "How are you?"},
        ],
        [
            {"role": "system", "content": "You are a witty comedian."},
            {"role": "user", "content": "Tell me a joke."},
        ]
    ]
    param: n: number of responses to generate for each message in messages_list
    """
    # If messages_list is not a list of list (i.e., only one conversation), convert it to a list of list
    assert isinstance(messages_list, list), "messages_list should be a list."
    try:
        if not isinstance(messages_list[0], list):
            messages_list = [messages_list]
    except:
        print(messages_list)
        raise IndexError("Something is wrong.")

    if len(messages_list) > 1:
        assert n == 1, "Currently, only n=1 is supported for multi-chat completion."

    num_workers = os.cpu_count()
    if "gpt" not in model:
        # Transform messages if n > 1
        messages_list *= n
        n = 1
        num_workers = 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        args = [(n, messages, model, temperature, isMultiLLM) for messages in messages_list]
        choices = executor.map(lambda p: chat_completion(*p), args)

    contents: list[str] = []
    for choice in choices:
        for c in choice:
            # print("CHOICE: ",c)
            contents.append(c.message.content)
    return contents

import requests
import time

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL")
print("OLLAMA URL:",OLLAMA_API_URL)
# OLLAMA_API_URL = "https://bfdb-34-23-111-246.ngrok-free.app/api/chat"  # Ollama runs locally by default
from types import SimpleNamespace
def dict_to_object(dictionary):
    return SimpleNamespace(**{k: dict_to_object(v) if isinstance(v, dict) else v for k, v in dictionary.items()})
import numpy as np
def chat_completion(n: int, messages: list[dict], model: str, temperature: float, isMultiLLM) -> list[dict]:
    """
    Generate n responses using the Ollama API.
    """
    responses = []  # Store multiple responses
    model = 'llama3'
    if (isMultiLLM):
        model = 'llama3' if np.random.rand() < 0.5 else 'qwen2.5-coder'
    
    for i in range(n):
            payload = {
                "model": model,
                "messages": messages,  # Extract latest user message
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }

            response = requests.post(OLLAMA_API_URL, json=payload)
            response_json = response.json()

            x = response_json['message']['content']
            responses.append(x)

    if not responses:
        print("Code terminated due to too many failed attempts!")

    print("HII", isMultiLLM)
    
    returnvalue = [{
            'message':{
                'role':'assistant',
                'content': x
            }
        } for x in responses]
    return [dict_to_object(x) for x in returnvalue]


def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end + 1])

    if code_string is None:
        return None
    # Add import statements if not present
    if "import" not in code_string:
        code_string = "import numpy as np\nimport random\nimport math\nimport scipy\nimport torch\n" + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name


def extract_to_hs(input_string: str):
    code_blocks = input_string.split("```python\n")[1:]

    try:
        parameter_ranges_block = "import numpy as np\n" + code_blocks[1].split("```")[0].strip()
        if any(keyword in parameter_ranges_block for keyword in ['inf', 'np.inf', 'None']):
            return None, None
        exec_globals = {}
        exec(parameter_ranges_block, exec_globals)
        parameter_ranges = exec_globals['parameter_ranges']
    except:
        return None, None

    function_block = code_blocks[0].split("```")[0].strip()

    paren_count = 0
    in_signature = False
    signature_start_index = None
    signature_end_index = None

    # Loop through the function block to find the start and end of the function signature
    for i, char in enumerate(function_block):
        if char == "d" and function_block[i:i + 3] == 'def':
            in_signature = True
            signature_start_index = i
        if in_signature:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if char == ':' and paren_count == 0:
                signature_end_index = i
                break

    if signature_start_index is not None and signature_end_index is not None:
        function_signature = function_block[signature_start_index:signature_end_index + 1]
        for param in parameter_ranges:
            pattern = rf"(\b{param}\b[^=]*=)[^,)]+"
            replacement = r"\1 {" + param + "}"
            function_signature = re.sub(pattern, replacement, function_signature, flags=re.DOTALL)
        function_block = function_block[:signature_start_index] + function_signature + function_block[
                                                                                       signature_end_index + 1:]

    return parameter_ranges, function_block


def format_messages(cfg, pre_messages):
    messages = [{"role": "system", "content": pre_messages["system"]},
                {"role": "user", "content": pre_messages["user"]}]
    return messages



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
    return (total_nodes/600, (complexity + 1)/15)


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

def get_behavior(code):
    return count_logical_nodes_and_complexity(remove_comments_and_docstrings(code))