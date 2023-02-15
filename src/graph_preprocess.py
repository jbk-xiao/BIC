import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import Data

path1 = Path(r'C:\Users\jbk-xiao\Documents\TwiBot-20\TwiBot-20-Format22\Twibot-20')
path2 = Path('data')
path3 = Path('src')

edge = pd.read_csv(path1 / 'edge.csv')
print('read edge.csv')
# node_des = np.load('data/node_des.npy', allow_pickle=True).tolist()
# labeled_user_count = len(node_des)
#%%
nodes = pd.read_json(path1 / 'node.json')
print('read node.json')
users = nodes[nodes.id.str.contains('^u')]

user_index_to_uid = list(users.id)
uid_to_user_index = dict(map(reversed, enumerate(user_index_to_uid)))
#%%
edge = edge[edge.relation != 'post']
edge = edge.reset_index(drop=True)
edge.source_id = list(map(lambda x: uid_to_user_index[x], edge.source_id))
edge.target_id = list(map(lambda x: uid_to_user_index[x], edge.target_id))

#%%
edge_index = []
source_id = torch.tensor(edge.source_id)
target_id = torch.tensor(edge.target_id)
a = torch.cat((source_id, target_id), dim=-1)
b = torch.cat((target_id, source_id), dim=-1)
edge_index = torch.stack((a, b))
#%%
torch.save(edge_index, 'data/edge_index.pt')
#%%

#%%

user_edge = {i: [] for i in range(len(users))}
for i in tqdm(range(len(edge.source_id))):
    user_edge[edge.source_id[i]].append([edge.source_id[i], edge.target_id[i]])
    user_edge[edge.source_id[i]].append([edge.target_id[i], edge.source_id[i]])
    user_edge[edge.target_id[i]].append([edge.source_id[i], edge.target_id[i]])
    user_edge[edge.target_id[i]].append([edge.target_id[i], edge.source_id[i]])
#%%
user_edge_new = {i: [] for i in range(11826)}

for i in tqdm(range(len(user_edge_new))):
    tmp = []
    for j in range(len(user_edge[i])):
        if user_edge[i][j] not in tmp:
            tmp.append(user_edge[i][j])
    tmp = np.array(tmp)
    user_edge_new[i] = tmp.transpose().tolist()
#%%

#%%
user_edge_final = {i:[] for i in range(len(user_edge_new))}
for i in tqdm(range(len(user_edge_new))):
    if user_edge_new[i] != []:
        user_order = list(set(user_edge_new[i][0]))
        for j in range(len(user_order)):
            if user_order[j] == i:
                user_order[0], user_order[j] = user_order[j], user_order[0]
        user_edge_final[i] = user_order
    else:
        user_order = [i]
        user_edge_final[i] = user_order
#%%
print(user_edge_final[0])
#%%
np.save('data/user_neighbor_index.npy', user_edge_final)
