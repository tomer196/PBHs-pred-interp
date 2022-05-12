import os

import torch

from torch import nn

from se3_transformer.equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from se3_transformer.equivariant_attention.fibers import Fiber
from data.aromatic_dataloader import AromaticDataset


class TFN(nn.Module):
    """SE(3) equivariant GCN"""
    def __init__(self, num_layers: int, atom_feature_size: int, 
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4, 
                 edge_dim: int=4, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels*num_degrees
        self.edge_dim = edge_dim

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, self.num_channels_out)}

        blocks = self._build_gcn(self.fibers, 1)
        self.block0, self.block1, self.block2 = blocks
        print(self.block0)
        print(self.block1)
        print(self.block2)

    def _build_gcn(self, fibers, out_dim):

        block0 = []
        fin = fibers['in']
        for i in range(self.num_layers-1):
            block0.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim))
            block0.append(GNormSE3(fibers['mid'], num_layers=self.num_nlayers))
            fin = fibers['mid']
        block0.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        block1 = [GMaxPooling()]

        block2 = []
        block2.append(nn.Linear(self.num_channels_out, self.num_channels_out))
        block2.append(nn.ReLU(inplace=True))
        block2.append(nn.Linear(self.num_channels_out, out_dim))

        return nn.ModuleList(block0), nn.ModuleList(block1), nn.ModuleList(block2)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.block0:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.block1:
            h = layer(h, G)

        for layer in self.block2:
            h = layer(h)

        return h


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int, node_feature_size: int,
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4,
                 edge_dim: int=4, div: float=4, pooling: str='avg', n_heads: int=1,
                 out_dim: int=1, **kwargs):
        super().__init__()
        # explainability
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None

        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads
        self.out_dim = out_dim

        self.fibers = {'in': Fiber(1, node_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees*self.num_channels)}

        blocks = self._build_gcn(self.fibers, out_dim)
        self.Gblock, self.FCblock = blocks
        # print(self.Gblock)
        # print(self.FCblock)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, 
                                  div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = []
        FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, graph, feat=None, return_attention=False):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(graph, self.num_degrees-1)
        if feat is None:
            feat = graph.ndata['f']
        feat.requires_grad = True
        self.input = feat
        # encoder (equivariant layers)
        h = {'0': feat}
        attn = []
        for layer in self.Gblock[:-2]:
            if isinstance(layer, GSE3Res):
                h, a = layer(h, G=graph, r=r, basis=basis)
                attn.append(a.detach())
            else:
                h = layer(h, G=graph, r=r, basis=basis)

        # save final conv for GradCAM
        with torch.enable_grad():
             h = self.Gblock[-2](h, G=graph, r=r, basis=basis)
             self.final_conv_acts = h['0']
        self.final_conv_acts.register_hook(self.activations_hook)
        h = self.Gblock[-1](h, G=graph, r=r, basis=basis)

        for layer in self.FCblock:
            h = layer(h)

        if return_attention:
            return h, attn
        else:
            return h


def create_model(args, dataset: AromaticDataset) -> nn.Module:
    num_node_features = dataset.num_node_features
    num_edge_features = dataset.num_edge_features
    num_targets = dataset.num_targets

    if args.model == 'SE3Transformer':
        model = SE3Transformer(
            num_layers=args.num_layers,
            node_feature_size=num_node_features,
            num_channels=args.num_channels,
            num_nlayers=args.num_nlayers,
            num_degrees=args.num_degrees,
            edge_dim=num_edge_features,
            div=args.div,
            pooling=args.pooling,
            n_heads=args.head,
            out_dim=num_targets
        )
    else:
        raise NotImplemented

    if args.restore is not None:
        path = os.path.join(args.exp_dir + '/model.pt')
        model.load_state_dict(torch.load(path))
    model.to(args.device)
    return model