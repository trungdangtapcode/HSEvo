import os
import re
import subprocess

# Path to your folder
# folder_path = "outputs/main/hsevo_knapsack_2025-04-07_03-09-14" #my best on hs
# folder_path = "outputs/main/hsevo_QDknapsack_2025-04-08_17-28-51" #suck on me
folder_path = "outputs/main/hsevo_QDknapsack_2025-04-08_18-41-04" #me good (17iter) on me, seem diver to top corner
folder_path = "outputs/main/hsevo_knapsack_2025-03-29_14-03-32" #kaggle best on me
folder_path = "outputs/main/hsevo_QDknapsack_2025-04-08_22-07-57" #mine (30iter) on me
folder_path = "outputs/main/hsevo_QDknapsack_2025-04-08_23-50-02" #mine (44iter) on me

# Regex pattern to extract iter and response numbers
pattern = re.compile(r"problem_iter(\d+)_response(\d+)\.txt")

# List and sort files based on iter and response numbers
files = []
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        iter_num = int(match.group(1))
        response_num = int(match.group(2))
        files.append((iter_num, response_num, filename))

# Sort by iter number, then by response number
files.sort()


from utils.archive import MAPElitesArchive
from utils.utils import *
# OUTPUT_PATH = "outputs/main/plots/offline"
OUTPUT_PATH = os.path.join(os.path.abspath(__file__),"..","outputs","plots","offline")


archive = MAPElitesArchive(2,10)

from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# os.mkdir(os.path.join(OUTPUT_PATH, current_time))
OUTPUT_PATH = os.path.join(OUTPUT_PATH, current_time)


def evaluate2(individual):
    filepath = list(os.path.join(folder_path, filename))
    filepath[-4:-3] = '.txt_stdout.'
    filepath = ''.join(filepath)
    with open(filepath, 'r') as f:
        stdout_str = f.read()
    # print(filepath)
    # print("STD:",stdout_str)
    # return
    traceback_msg = filter_traceback(stdout_str)

    if (traceback_msg == ''): 
        try:
            individual['obj'] = -float(stdout_str.split('\n')[-2])
            individual["exec_success"] = True
        except:
            pass
def evaluate(individual):
    inner_run = run_code(individual)
    
    if (inner_run is None):
        return
    try:
        inner_run.communicate(timeout=120)
    except subprocess.TimeoutExpired as e:
        inner_run.kill()
        inner_run.wait()
        print("Timeout")
        return
    
    stdout_filepath = "trash.txt"
    with open(stdout_filepath, 'r') as f:  # read the stdout file
        stdout_str = f.read()

    traceback_msg = filter_traceback(stdout_str)

    if traceback_msg == '': 
        individual['obj'] = -float(stdout_str.split('\n')[-2])
        individual["exec_success"] = True
    else:
        # print("Code traceback:",traceback_msg)
        individual['obj'] = float('inf')
        print("Code execution failed!")
        return

        
def run_code(individual: dict) -> subprocess.Popen:
    """
    Write code into a file and run eval script.
    """
    print('[*] running code')

    with open("problems/QDknapsack/gpt.py", 'w') as file:
        file.writelines(individual["code"] + '\n')

    # Execute the python file with flags
    # with open(individual["stdout_filepath"], 'w') as f:
    with open("trash.txt", 'w') as f:
        eval_file_path = f'problems/QDknapsack/eval.py' 
        process = subprocess.Popen(['python', '-u', eval_file_path, "1", "t", "train"],
                                    stdout=f, stderr=f)



    return process


# Loop through sorted files

cnt = 0

for iter_num, response_num, filename in files[:]:
    # print(f"Processing: {filename} (iter={iter_num}, response={response_num})")
    filepath = os.path.join(folder_path, filename)
    with open(os.path.join(folder_path, filename), 'r') as file:
        content = file.read()
        # Do something with the content
        
    code = extract_code_from_generator(content)
    if (code is None): continue
    
    behavior = get_behavior(code)

    individual = {
        "file_name": filename,
        "code": code,
        "obj": float('inf'),
        "exec_success": False,
    }
    evaluate2(individual)
    cnt += 1
    archive.add(individual, individual['obj'], behavior)
    # archive.save_img(title=f"num individuals: {cnt}", folder_savepath=OUTPUT_PATH)
    

    print(individual["obj"], behavior)
    # break

z = 0
for i in range(len(archive.get_elites())):
    if (archive.get_elites()[i][1]<0):
        print(archive.get_elites()[i][1],'day ne')
        z = i
        break

z = -1
print(z)
print('fitness: ',archive.get_elites()[z][1])
print('behavior: ',archive.get_elites()[z][2])
print(archive.get_elites()[z][0]['code'])


