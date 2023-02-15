import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

path1 = Path(r'C:\Users\jbk-xiao\Documents\TwiBot-20\TwiBot-20-Format22\Twibot-20')
path2 = Path('data')
path3 = Path('src')

#%%
def str_to_bool(label):
    if label[1] == 'human':
        return (label[0], 1)
    else:
        return (label[0], 0)
#%%

node_info = pd.read_json(path1 / 'node.json')
label = pd.read_csv(path1 / 'label.csv')
split = pd.read_csv(path1 / 'split.csv')
node_info = pd.merge(node_info, label)
node_info = pd.merge(node_info, split)

#%%

node_des = node_info['description']
node_name = node_info['username']
node_label = node_info['label']
node_split = node_info['split']

node_des = dict(map(lambda x: x, enumerate(node_des)))
node_name = dict(map(lambda x: x, enumerate(node_name)))
node_label = dict(map(str_to_bool, enumerate(node_label)))
node_split = dict(map(lambda x: x, enumerate(node_split)))

np.save('data/node_des.npy', node_des)
np.save('data/node_name.npy', node_name)
np.save('data/node_label.npy', node_label)
np.save('data/node_split.npy', node_split)
