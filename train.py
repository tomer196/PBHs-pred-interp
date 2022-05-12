import json
import random
from datetime import datetime
from time import time
import os
import sys
import warnings
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.plotting import plot_compare
from data.aromatic_dataloader import create_data_loaders
from se3_transformer.models import create_model

from utils.args import Args

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch

from torch import optim

def to_np(x):
    return x.cpu().detach().numpy()

def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, scheduler, args, writer):
    model.train()

    start_time = time()
    num_iters = len(dataloader)
    train_loss = []
    rloss = []
    with tqdm(dataloader, unit="batch") as tepoch:
        for i, (g, y) in enumerate(tepoch):
            g = g.to(args.device)
            y = y.to(args.device)

            # run model forward and compute loss
            pred = model(g)
            l1_loss, __, rl = loss_fnc(pred, y, dataloader.dataset)

            # backprop
            optimizer.zero_grad()
            l1_loss.backward()
            optimizer.step()

            train_loss.append(l1_loss.item())
            rloss.append(rl.item())
            # if (i+1) % args.print_interval == 0:
            #     print(f"[{epoch}|{i}] l1 loss: {np.mean(train_loss):.4f}, "
            #           f"rescale loss: {np.mean(rloss):.4f}")

            scheduler.step(epoch + i / num_iters)
            tepoch.set_postfix(loss=np.mean(train_loss).item())
    print(f"[{epoch}|train] l1 loss: {np.mean(train_loss):.4f}+-{np.std(train_loss):.4f}, "
          f"rescale loss: {np.mean(rloss):.4f}+-{np.std(rloss):.4f},"
                  f" in {int(time()-start_time)} secs")
    writer.add_scalar('Train L1', np.mean(train_loss), epoch)
    writer.add_scalar('Train L1 scaled', np.mean(rloss), epoch)

def val_epoch(epoch, model, loss_fnc, dataloader, args, writer):
    model.eval()
    val_losses = []
    rloss = []
    for i, (g, y) in enumerate(dataloader):
        g = g.to(args.device)
        y = y.to(args.device)

        # run model forward and compute loss
        with torch.no_grad():
            pred = model(g)
        l1, __, rl = loss_fnc(pred, y, dataloader.dataset)
        rloss.append(rl.item())
        val_losses.append(l1.item())

    print(f"...[{epoch}|val] l1 loss: {np.mean(val_losses):.4f}+-{np.std(val_losses):.4f}, "
          f"rescale loss: {np.mean(rloss):.4f}+-{np.std(rloss):.4f}")
    writer.add_scalar('Val L1', np.mean(val_losses), epoch)
    writer.add_scalar('Val L1 scaled', np.mean(rloss), epoch)
    return np.mean(val_losses)

def test_epoch(epoch, model, loss_fnc, dataloader, args):
    model.eval()
    test_losses = []
    pred_list = []
    gt_list = []
    rloss = []
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

    print(
        f"...[{epoch}|test] l1 loss: {np.mean(test_losses):.4f}+-{np.std(test_losses):.4f}, "
        f"rescale loss: {np.mean(rloss):.4f}+-{np.std(rloss):.4f}")

    pred_list = np.concatenate(pred_list) * dataloader.dataset.std.numpy()+dataloader.dataset.mean.numpy()
    gt_list = np.concatenate(gt_list) * dataloader.dataset.std.numpy()+dataloader.dataset.mean.numpy()
    tasks = args.target_features.replace(' ', '').split(',')
    pred_list = pred_list.reshape(-1, len(tasks))
    for i in range(len(tasks)):
        plot_compare(gt=gt_list[:, i], pred=pred_list[:, i],
                     title=f'{args.name}-{tasks[i]}', args=args)
        print(f'{tasks[i]}, MAE: {np.abs(gt_list[:, i] - pred_list[:, i]).mean()}, '
              f'MSE: {((gt_list[:, i] - pred_list[:, i]) ** 2).mean()}')

    return np.mean(rloss)

# Loss function
def task_loss(pred, target, dataset):
    l1_loss = torch.mean(torch.abs(pred - target))
    l2_loss = torch.mean((pred - target)**2)
    rescale_loss = dataset.rescale_loss(l1_loss)
    return l1_loss, l2_loss, rescale_loss

def main(args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(args)

    # Choose model
    model = create_model(args, train_loader.dataset)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                               args.num_epochs,
                                                               eta_min=1e-4)

    # Save path
    writer = SummaryWriter(log_dir=args.exp_dir)

    # Run training
    print('Begin training')
    best_val_loss = 1e9
    best_epoch = 0
    for epoch in range(args.num_epochs):

        train_epoch(epoch, model, task_loss, train_loader, optimizer, scheduler, args, writer)
        val_loss = val_epoch(epoch, model, task_loss, val_loader, args, writer)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.exp_dir + '/model.pt')

    print(f'{best_epoch=}, {best_val_loss=:.4f}')
    _ = test_epoch(epoch, model, task_loss, test_loader, args)
    writer.close()

if __name__ == '__main__':
    args = Args().parse_args()

    # Fix name
    if not args.name:
        graph_type = 'rings' if args.rings_graph else 'atoms'
        time_str = datetime.now().strftime("%H:%M:%S_%d-%m-%y")
        args.name = f'{args.model}-{graph_type}' # -{time_str}'
    print(args.name)
    args.exp_dir = f'{args.save_dir}/{args.name}'

    # Create model directory
    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    with open(args.exp_dir + '/args.txt', "w") as f:
        json.dump(args.__dict__, f, indent=2)
    # Automatically choose GPU if available
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("\n\nArgs:", args)

    # Where the magic is
    main(args)
