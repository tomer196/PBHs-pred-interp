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
    names = [
        # linear
        'hc_c34h20_0pent_442', 'hc_c38h22_0pent_1677', 'hc_c34h20_0pent_434', 'hc_c34h20_0pent_425',
        # angular
        'hc_c34h20_0pent_62', 'hc_c34h20_0pent_29', 'hc_c34h20_0pent_33', 'hc_c34h20_0pent_41',
        'hc_c42h24_0pent_950', 'hc_c38h22_0pent_1053', 'hc_c38h22_0pent_1046',
        # Branched
        'hc_c38h22_0pent_159', 'hc_c34h20_0pent_193', 'hc_c30h18_0pent_56', 'hc_c30h18_0pent_57',
        # 2 sample
        # 'hc_c38h22_0pent_1009', 'hc_c34h20_0pent_274', 'hc_c34h20_0pent_177',
        # 'hc_c38h22_0pent_1295', 'hc_c34h20_0pent_337', 'hc_c34h20_0pent_183',
        # # old slide
        # 'hc_c34h20_0pent_123', 'hc_c34h20_0pent_121', 'hc_c42h24_0pent_103',
        # 'hc_c42h24_0pent_1015', 'hc_c38h22_0pent_105', 'hc_c38h22_0pent_100',
        # # cove
        # 'hc_c38h22_0pent_170', 'hc_c38h22_0pent_171', 'hc_c38h22_0pent_576',
        # # fjord
        # 'hc_c34h20_0pent_14', 'hc_c38h22_0pent_55', 'hc_c38h22_0pent_575',
        # # helix
        # 'hc_c42h24_0pent_110',  'hc_c38h22_0pent_360', 'hc_c38h22_0pent_408'
    ]

    # names = ['hc_c42h24_0pent_4377']

    dir_name = 'inter-GAP-final'
    try_mkdir(dir_name)
    df = dataloader.dataset.df_all
    for i, name in enumerate(names):
        if os.path.isfile(f'{dir_name}/{args.target_features}-{name}.png'):
            print(i)
            continue
        else:
            df_row = df[df.name == name].iloc[0]
            mol, edges, name = dataloader.dataset.get_mol(df_row)
            title = f'{args.target_features}-{name}'
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

    # args.name='Erel'
    args.name = 'SE3-knots-GAP_eV'
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
