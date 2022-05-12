import json
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.aromatic_dataloader import create_data_loaders
from gradram import plot_mol_gradram, grad_ram
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
    samples = range(len(dataloader.dataset.df_all))
    samples=np.random.permutation(samples)

    for i in samples:
        df_row = dataloader.dataset.df_all.iloc[i]
        mol, edges, name = dataloader.dataset.get_mol(df_row)
        pdf_filename = f'{args.exp_dir}/interp-{args.target_features}/{args.target_features}-{name}.pdf'
        if os.path.isfile(pdf_filename):
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
            pred.backward()

            final_conv_acts = model.final_conv_acts.view(-1, 96)
            final_conv_grads = model.final_conv_grads.view(-1, 96)
            grad_ram_weights = grad_ram(final_conv_acts, final_conv_grads, False)
            fig = plot_mol_gradram(g, mol, edges, grad_ram_weights, y.item(), args.target_features)
            fig.savefig(pdf_filename, bbox_inches='tight')
            # fig.show()
            plt.close(fig)

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
