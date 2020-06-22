import matplotlib.pyplot as plt
import os
import glob
import ast
import pandas as pd
import seaborn as sns

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

dataset = []
for optimizer in experiments.keys():
    raw_data = experiments[optimizer]
    for size in ['25', '50', '125', '250', '500', '800']:
        hidden_size = 'hidden' + size
        if hidden_size in raw_data.keys():
            dataset = dataset + [
                [
                    optimizer,
                    hidden_size,
                    raw_data[hidden_size]['sample_num'][i],
                    raw_data[hidden_size]['rmsd_avg'][i],
                    raw_data[hidden_size]['drmsd_avg'][i],
                    raw_data[hidden_size]['train_loss_values'][i],
                    raw_data[hidden_size]['validation_loss_values'][i]
                ]
                for i in range(len(raw_data[hidden_size]['sample_num']))
            ]

df = pd.DataFrame(data=dataset,
                  columns=['optimizer', 'hidden_size', 'sample_num', 'rmsd_avg', 'drmsd_avg', 'train_loss_values',
                           'validation_loss_values'])


def showGraph(optimizer, y):
    sns.relplot(x='sample_num', y=y, kind='line', hue='optimizer', data=df[df.hidden_size.eq('hidden25')])
    g = sns.FacetGrid(df[df.optimizer.eq('adam')], col_wrap=2, col='hidden_size')
    g.map(plt.plot, 'sample_num', y)
    plt.show()
