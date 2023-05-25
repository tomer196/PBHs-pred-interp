import sys
import random
from time import time
from typing import Tuple

import dgl
import networkx as nx
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot

from data.mol import load_xyz, Mol
from utils.args import Args
from utils.knotgraph import get_knots, get_knots_connectivity
from utils.molgraph import get_connectivity_matrix, get_edges
from utils.plotting import get_figure, plot_graph, moldraw

DTYPE = torch.float32
INT_DTYPE = torch.int8
# ATOMS_LIST = __ATOM_LIST__[:8]
ATOMS_LIST = ['H', 'C']
KNOTS_LIST = ['bn']

def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(y, dim=0)

class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = torch.randn(3, 3)
        Q, __ = torch.linalg.qr(M)
        return x @ Q

class AromaticDataset(Dataset):
    def __init__(self, args, task: str = 'train'):
        """
        Args:
            args:
            task: Select the dataset to load from (train/val/test).
        """
        self.csv_file, self.xyz_root = args.csv_file, args.xyz_root

        self.task = task
        self.target_features = args.target_features.replace(' ', '').split(',')
        self.rings_graph = args.rings_graph
        self.transform = RandomRotation() if (args.transform and task =='train') else None
        self.normalize = args.normalize

        self.df = getattr(args, task + '_df').reset_index()
        if args.normalize:
            train_df = args.train_df
            target_data = train_df[self.target_features].values
            self.mean = torch.tensor(target_data.mean(0), dtype=DTYPE)[None, :]
            self.std = torch.tensor(target_data.std(0), dtype=DTYPE)[None, :]
        else:
            self.std = torch.ones(1, dtype=DTYPE)
            self.mean = torch.zeros(1, dtype=DTYPE)

        self.examples = np.arange(self.df.shape[0])
        if args.sample_rate < 1:
            random.shuffle(self.examples)
            num_files = round(len(self.examples) * args.sample_rate)
            self.examples = self.examples[:num_files]

        G, y = self.__getitem__(0)
        self.num_targets = y.shape[1]
        self.num_node_features = G.ndata['f'].shape[1]
        self.num_edge_features = G.edata['w'].shape[1]

        self.df_all = args.df_all.reset_index()

    def __len__(self):
        return len(self.examples)

    def rescale_loss(self, x):
        # Convert from normalized to the original representation
        if self.normalize:
            x = x * self.std.to(x.device).mean()
        return x

    def get_mol(self, df_row) -> Tuple[Mol, list, str]:
        try:
            name = df_row['name']
        except:
            name = df_row['molecule']
        xyz_file = self.xyz_root + name + '.xyz'

        mol = load_xyz(xyz_file)
        atom_connectivity = get_connectivity_matrix(mol.atoms, skip_hydrogen=False)  # build connectivity matrix
        edges = get_edges(atom_connectivity)  # edges = bonds
        return  mol, edges, name

    def get_all(self, df_row):
        # extract targets
        y = torch.tensor(df_row[self.target_features].values.astype(np.float32),
                         dtype=DTYPE)[None, :]

        mol, edges, _ = self.get_mol(df_row)
        # get_figure(mol, edges, showPlot=True)

        # creation of nodes, edges and there features
        if self.rings_graph:
            preprocessed_path = self.xyz_root + "_rings_preprocessed/" + name + ".xyz"
            try_mkdir(self.xyz_root + "_rings_preprocessed/")
            if Path(preprocessed_path).is_file():
                x, edges, nodes_features, edge_features = torch.load(preprocessed_path)
            else:
                mol, edges, _ = self.get_mol(df_row)
                mol_graph = nx.Graph(edges)
                knots = get_knots(mol.atoms, mol_graph)
                edges = get_knots_connectivity(knots)
                x = torch.tensor([k.get_coord() for k in knots], dtype=DTYPE)
                knot_type = torch.tensor([KNOTS_LIST.index(k.cycle_type) for k in knots]).unsqueeze(1)
                nodes_features = one_hot(knot_type, num_classes=len(KNOTS_LIST)).permute(0, 2, 1).float()
                edges = torch.tensor(edges, dtype=DTYPE).view(-1, 2)
                edges = torch.cat([edges, edges[:, [1, 0]]], 0)
                if edges.shape[0] == 0:
                    edges = torch.zeros(1, 2)
                edge_features = torch.ones(edges.shape[0], 1)  # null features
                torch.save([x, edges, nodes_features, edge_features], preprocessed_path)

        else:  # atoms graph
            x = torch.tensor([a.get_coord() for a in mol.atoms], dtype=DTYPE)
            atom_element = torch.tensor([ATOMS_LIST.index(atom.element) for atom in mol.atoms]).unsqueeze(1)
            nodes_features = one_hot(atom_element, num_classes=len(ATOMS_LIST)).permute(0, 2, 1).float()
            edges = torch.tensor(edges, dtype=DTYPE)
            edges = torch.cat([edges, edges[:, [1, 0]]], 0)
            edge_features = torch.ones(edges.shape[0], 1)  # null features

        # Load target
        if self.normalize:
            y = (y - self.mean) / self.std

        # Augmentation on the coordinates
        if self.transform:
            x = self.transform(x)

        # Create graph
        G = dgl.graph((edges[:, 0].long(), edges[:, 1].long()))

        # Add node features to graph
        G.ndata['x'] = x  # [num_nodes(atoms/knots), 3]
        G.ndata['f'] = nodes_features  # [num_nodes, #node_features, 1]

        # Add edge features to graph
        G.edata['d'] = x[edges[:, 1].long()] - x[edges[:, 0].long()]  # [num_edges, 3]
        G.edata['w'] = edge_features  # [num_edges, #edge_features]

        # plot_graph(G)
        return G, y

    def unnormalize(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def __getitem__(self, idx):
        index = self.examples[idx]
        df_row = self.df.loc[index]
        return self.get_all(df_row)

def get_splits(args, random_seed = 42, val_frac = 0.1, test_frac = 0.1):
    np.random.seed(seed=random_seed)
    df = pd.read_csv(args.csv_file)

    df_all = df.copy()
    df_test = df.sample(frac=test_frac, random_state=random_seed)
    df = df.drop(df_test.index)
    df_val = df.sample(frac=val_frac, random_state=random_seed)
    df_train = df.drop(df_val.index)
    return df_train, df_val, df_test, df_all

def create_data_loaders(args):
    args.train_df, args.val_df, args.test_df, args.df_all = get_splits(args)

    train_dataset = AromaticDataset(
        args=args,
        task='train',
    )
    val_dataset = AromaticDataset(
        args=args,
        task='val',
    )
    test_dataset = AromaticDataset(
        args=args,
        task='test',
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate,
                              num_workers=args.num_workers,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=collate,
                            num_workers=args.num_workers,
                            pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate,
                             num_workers=args.num_workers,
                             pin_memory=True)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    args = Args().parse_args()

    dataset = AromaticDataset(
        args=args,
        task='train',
    )
    import matplotlib.pyplot as plt
    for i in range(5):
        mol, edges = dataset.get_mol(i)
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        ax.axis('off')

        # plot molecule
        moldraw(ax, mol, edges)
        ax.set_title()
        fig.show()
    s=time()
    print(dataset[0])
    print(time()-s)
    dataloader = DataLoader(dataset,
                            batch_size=32,
                            num_workers=8,
                            shuffle=True,
                            collate_fn=collate)

    times = []
    s=time()
    for i, data in enumerate(dataloader):
        times.append(time()-s)
        # print("MINIBATCH")
        # print(data)
        s=time()
        if i==20:
            print(np.mean(times), np.std(times))
            sys.exit()
    print(np.mean(times), np.std(times))

