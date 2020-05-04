import matplotlib.pyplot as plt
import os
import glob
import ast

experiments = {}
output_dir = os.curdir + '/output'
file_names = glob.glob(output_dir + "/*MB32.txt")
for file_name in file_names:
    split = file_name.split('-')
    optimizer = split[5]
    hidden_size = split[6]
    dataset = split[7]
    if split[8] == 'restart':
        optimizer = 'sgdr'
    result = ''
    with open(file_name, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.find('drmsd') > 0:
                result = line

    str_result = "{" + result.split("n', ")[-1]
    obj_result = ast.literal_eval(str_result)
    if optimizer not in experiments.keys():
        experiments[optimizer] = {}
    if hidden_size not in experiments[optimizer].keys():
        experiments[optimizer][hidden_size] = {}

    experiments[optimizer][hidden_size] = obj_result

print(experiments)

