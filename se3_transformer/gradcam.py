import pickle

import dgl
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.preprocessing import MinMaxScaler
from torch import nn, Tensor

from data.mol import Mol
from utils.plotting import plot_graph_weight, plot_attention


def grad_cam(final_conv_acts, final_conv_grads):
    # print('grad_cam')
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    scaled_grad_cam_weights = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        np.array(node_heat_map).reshape(-1, 1)).reshape(-1, )
    return scaled_grad_cam_weights

def grad_ram(final_conv_acts, final_conv_grads, normalize=True):
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = (alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    node_heat_map = np.array(node_heat_map)
    if normalize:
        node_heat_map = node_heat_map / np.abs(node_heat_map).max()
    return node_heat_map

def plot_explain(model: nn.Module, g: dgl.graph, mol: Mol , edges: list,
                 gt: Tensor, pred: Tensor,
                 title: str= '', dir_name: str = 'inter', normalize=True):
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(1, 1, figsize=(7.5, 3))
    # Attention vizualization
    # with torch.no_grad():
    #     _, attn = model(g, return_attention=True)
    # plot_attention(g, mol, edges, attn, 'Attention', axes[0])

    if pred.shape[0] != 1:
        raise Exception('GradRAM model can be used only with single predicted output')
    pred.backward()

    final_conv_acts = model.final_conv_acts.view(-1, 96)
    final_conv_grads = model.final_conv_grads.view(-1, 96)
    if 'c26' in title:
        size=2500
    elif 'c30' in title:
        size=2500
    elif 'c34' in title:
        size=2500
    elif 'c38' in title:
        size=2500
    elif 'c42' in title:
        size=1800
    w = grad_ram(final_conv_acts, final_conv_grads, normalize)
    plot_graph_weight(g, mol, edges,
                      w,
                      'GradRAM', axes, size=size)
    # fig.suptitle(f'{title} \n'
    #              f'gt: {gt.item():.3f}, pred: {pred.item():.3f}')
    property = title.split('_')[0]
    fig.suptitle(f'{property}: {gt.item():.3f}')
    fig.savefig(f'{dir_name}/{title}.png')
    # fig.show()
    plt.close(fig)
    with open(f'{dir_name}/{title}.pkl', 'wb') as f:
        pickle.dump({'g': g, 'mol': mol, 'edges': edges, 'w': w,
                     'property': property, 'value': gt.item()}, f)

    # with open('saved_dictionary.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)


