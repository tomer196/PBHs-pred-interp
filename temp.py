import json
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

from data.aromatic_dataloader import create_data_loaders
from gradram import plot_mol_gradram
from se3_transformer.models import create_model
from se3_transformer.utils.utils_logging import try_mkdir
from utils.args import Args


#%%
saved_exp_name = 'Erel'
args = Args().parse_args()
args.name = saved_exp_name
print(f'Loading the arguments for experiment named: {args.name}')

args.exp_dir = f'{args.save_dir}/{args.name}'
with open(args.exp_dir + '/args.txt', "r") as f:
    args.__dict__ = json.load(f)
args.restore = True

# Automatically choose GPU if available
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader, _, _ = create_data_loaders(args)
dataset = train_loader.dataset
model = create_model(args, dataset)

#%%
def grad_ram(final_conv_acts, final_conv_grads):
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = (alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    return np.array(node_heat_map)

#%%
name = 'hc_c38h22_0pent_159'
print(f'Visualizing IVs for molecule: {name}')

model.eval()
df = dataset.df_all
df_row = df[df.molecule == name].iloc[0]
mol, edges, name = dataset.get_mol(df_row)
title = f'{args.target_features}-{name}'

g, y = dataset.get_all(df_row)
pred = model(g.to(args.device))

y = dataset.unnormalize(y)
pred = dataset.unnormalize(pred)
pred.backward()

final_conv_acts = model.final_conv_acts.view(-1, 96)
final_conv_grads = model.final_conv_grads.view(-1, 96)
grad_ram_weights = grad_ram(final_conv_acts, final_conv_grads)
fig = plot_mol_gradram(g, mol, edges, grad_ram_weights, y.item(), args.target_features)
fig.show()


