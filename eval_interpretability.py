import json
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.aromatic_dataloader import create_data_loaders
from se3_transformer.gradcam import plot_explain
from se3_transformer.models import create_model

import warnings

from se3_transformer.utils.utils_logging import try_mkdir
from utils.args import Args

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch


def to_np(x):
    return x.cpu().detach().numpy()

def interpretation(model, dataloader, args):
    model.eval()
    if not hasattr(args, 'dataset') or args.dataset == 'pahs-8606':
        samples = [0, 146, 625]
        # samples = [0]
        samples = range(len(dataloader.dataset.df_all))
        samples=np.random.permutation(samples)
        # df = dataloader.dataset.df_all
        # samples = np.argsort(-df[args.target_features].values)[:50]
    elif args.dataset == '49002-peri':
        samples = [0, 1116, 3068]
        # samples = [58, 3197,  522, 2135, 3068]  # outliers
    else:
        raise NotImplemented

    dir_name = 'inter-Erel-normalize'
    try_mkdir(dir_name)
    for i in samples:
        df_row = dataloader.dataset.df_all.iloc[i]
        mol, edges, name = dataloader.dataset.get_mol(df_row)
        title = f'{args.target_features}-{name}'
        if os.path.isfile(f'{dir_name}/{args.target_features}-{name}.png'):
            print(i)
            continue
        else:
            print(i, name)
            g, y = dataloader.dataset.get_all(df_row)
            g = g.to(args.device)
            y = y.to(args.device)

            pred = model(g)

            y = y.cpu() * dataloader.dataset.std + dataloader.dataset.mean
            pred = pred.cpu() * dataloader.dataset.std + dataloader.dataset.mean
            plot_explain(model, g, mol, edges, y, pred, title, dir_name, False)

def main(args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(args)

    # Choose model
    if not args.restore:
        print("FLAGS.restore must be set")
    model = create_model(args, val_loader.dataset)

    # Run training
    print('Begin evaluation')
    interpretation(model, train_loader, args)

if __name__ == '__main__':
    args = Args().parse_args()

    # torch.manual_seed(0)
    # np.random.seed(0)

    args.name='Erel'
    # args.name = 'SE3-knots-GAP_eV'
    print(args.name)
    args.exp_dir = f'{args.save_dir}/{args.name}'
    with open(args.exp_dir + '/args.txt', "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True
    args.transform = False
    # Automatically choose GPU if available
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("\n\nArgs:", args)

    main(args)
