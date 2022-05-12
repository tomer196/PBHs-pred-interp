import json

from data.aromatic_dataloader import create_data_loaders
from se3_transformer.models import create_model
from train import task_loss

import dgl
import warnings

from utils.args import Args
from utils.plotting import plot_compare

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch


def to_np(x):
    return x.cpu().detach().numpy()

def test_epoch(epoch, model, loss_fnc, dataloader, args):
    model.eval()
    test_losses = []
    pred_list = []
    gt_list = []
    rloss = []
    gs = []
    for i, (g, y) in enumerate(dataloader):
        g = g.to(args.device)
        y = y.to(args.device)

        # run model forward and compute loss
        with torch.no_grad():
            pred = model(g)
        l1, __, rl = loss_fnc(pred, y, dataloader.dataset)
        rloss.append(rl.item())
        test_losses.append(l1.item())
        pred_list.append(pred.cpu().numpy().squeeze())
        gt_list.append(y.cpu().numpy())
        gs += dgl.unbatch(g.to('cpu'))

    print(
        f"...[{epoch}|test] l1 loss: {np.mean(test_losses):.4f}+-{np.std(test_losses):.4f}, "
        f"rescale loss: {np.mean(rloss):.4f}+-{np.std(rloss):.4f}")

    pred_list = np.concatenate(pred_list) * dataloader.dataset.std.numpy() + dataloader.dataset.mean.numpy()
    gt_list = np.concatenate(gt_list) * dataloader.dataset.std.numpy() + dataloader.dataset.mean.numpy()
    tasks = args.target_features.replace(' ', '').split(',')
    pred_list = pred_list.reshape(-1, len(tasks))
    for i in range(len(tasks)):
        plot_compare(gt=gt_list[:,i], pred=pred_list[:,i],
                     title=f'{args.name}-{tasks[i]}', args=args)
        print(f'{tasks[i]}, MAE: {np.abs(gt_list[:,i]-pred_list[:,i]).mean()}, '
              f'MSE: {((gt_list[:,i]-pred_list[:,i])**2).mean()}')
    rel_error = np.abs(pred_list[:,0]-gt_list[:, 0])/pred_list[:,0]
    print(f'Relative error: {np.mean(rel_error):.4f}+-{np.std(rel_error):.4f}')
    return np.mean(rloss)

def main(args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(args)

    # Choose model
    if not args.restore:
        print("FLAGS.restore must be set")
    model = create_model(args, train_loader.dataset)

    # Run training
    print('Begin evaluation')
    test_epoch(0, model, task_loss, test_loader, args)

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    args = Args().parse_args()
    args.name = 'Erel'

    args.exp_dir = f'{args.save_dir}/{args.name}'

    with open(args.exp_dir + '/args.txt', "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True
    # Automatically choose GPU if available
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(args.name)
    print("\n\nArgs:", args)

    main(args)
