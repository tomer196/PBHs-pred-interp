import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
import networkx as nx

def get_figure(_molrepr, _edges, filename=None, showPlot=False):
    """
    get_figure(_molrepr: Mol Object, _egdes: list, filename: string)

    Function that draws the molecule and saves and/or shows it.

    in:
    _molrepr: Mol Object containing XYZ data.
    _edges: list of tuples containing the indices of 2 atoms bonding.
    filename: name of the file where the figure will be saved.

    """

    # set parameters
    plt.rcParams.update({'font.size': 22})

    fig, ax = plt.subplots(1, 1, figsize=(10,12))

    # plot scan and molecule with path
    moldraw(ax, _molrepr, _edges)

    # save figure
    if filename:
        fig.savefig(filename)

    # show figure
    if showPlot:
        plt.show()

    return None

def moldraw(ax, _molrepr, _edges, plot_h=False):
    """
    moldraw(ax, _molrepr: Mol Object, _edges: list)

    Function that draws the molecule.

    in:
    _molrepr: Mol Object containing XYZ data.
    _edges: list of tuples containing the indices of 2 atoms bonding.

    """

    atom_colors = {'H':'silver','N':'blue','O':'red','S':'goldenrod','B':'green'}

    # set aspect of subplot
    ax.set_aspect('equal')
    ax.axis('off')

    # plot molecule
    for edge in _edges:
        bond = []
        Hbond = False
        for atom_idx in edge:
            for atom in _molrepr.atoms:
                if atom_idx == atom.index:
                    bond.append(atom)
                    if atom.element != 'C':
                        if atom.element == 'BH':
                            atom.element = 'B'
                        elif atom.element == 'H':
                            Hbond = True
                        if Hbond and not plot_h:
                            continue
                        ax.text(atom.x, atom.y, atom.element, ha='center', va='center', color=atom_colors[atom.element],
                                zorder=2, bbox=dict(facecolor='white', edgecolor='none', boxstyle='circle, pad=0.1'))
        x = [atom.x for atom in bond]
        y = [atom.y for atom in bond]
        if Hbond:
            if plot_h:
                ax.plot(x, y, c=atom_colors['H'], linestyle='-', zorder=0)
        else:
            ax.plot(x, y, c='black', linestyle='-', zorder=1)

    return None

def plot_compare(pred, gt, args, title=''):
    # gt = np.concatenate([gt[:1405], gt[1406:]])
    # pred = np.concatenate([pred[:1405], pred[1406:]])
    min_val = np.concatenate([gt, pred]).min()
    max_val = np.concatenate([gt, pred]).max()
    plt.scatter(gt, pred)
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.title(f'{title}, MAE: {np.abs(gt-pred).mean():.4f}+-{np.abs(gt-pred).std():.4f}')
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(args.exp_dir + '/' + title.replace('/','') + '.png')
    plt.show()

def align_to_xy_plane(x):
    """
    Rotate the molecule into xy-plane.

    """
    I = np.zeros((3,3)) # set up inertia tensor I
    com = np.zeros(3) # set up center of mass com

    # calculate moment of inertia tensor I
    for i in range(x.shape[0]):
        atom = x[i]
        I += np.array([[(atom[1]**2+atom[2]**2),-atom[0]*atom[1]       ,-atom[0]*atom[2]],
                        [-atom[0]*atom[1]       ,(atom[0]**2+atom[2]**2),-atom[1]*atom[2]],
                        [-atom[0]*atom[2]       ,-atom[1]*atom[2]       ,atom[0]**2+atom[1]**2]])

        com += atom
    com = com/len(com)

    # extract eigenvalues and eigenvectors for I
    # np.linalg.eigh(I)[0] are eigenValues, [1] are eigenVectors
    eigenVectors = np.linalg.eigh(I)[1]
    eigenVectorsTransposed = np.transpose(eigenVectors)

    a = []
    for i in range(x.shape[0]):
        xyz = x[i]
        a.append(np.dot(eigenVectorsTransposed, xyz - com))
    return np.stack(a)

def align_to_x_plane(x):
    """
    Rotate the molecule into x axis.
    """
    x = x[:,:2]
    xm = x.mean(0)
    xc = x - xm
    _, _, Vt = np.linalg.svd(xc)
    xr = xc @ Vt.T
    return xr, xm, Vt

def plot_graph(g, title='', ax=None):
    if ax is None:
        ax = plt.gca()
    x = g.ndata['x'].detach().cpu().numpy()
    # scatter3d(x[:, 0], x[:, 1], x[:, 2])
    x = align_to_xy_plane(x)
    #scatter(x[:, 0], x[:, 1])
    edges = torch.stack(g.edges(), dim=1).detach().cpu().numpy()
    # remove double edges
    n = int(edges.shape[0]/2)
    edges = edges[:n]

    # if we want later we can add node/edge type (extract from the features)
    plt.rcParams.update({'font.size': 16})
    ax.set_aspect('equal')
    ax.axis('off')
    ax.scatter(x[:, 0], x[:, 1])
    for edge_ind in range(edges.shape[0]):
        edge = edges[edge_ind]
        ax.plot(x[edge, 0], x[edge, 1], c='black')
    ax.set_title(f'{title}')
    plt.show()
    return ax
