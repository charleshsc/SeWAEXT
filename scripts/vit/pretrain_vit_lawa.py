import argparse
import os
import time
import random
import numpy as np
import copy
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from src.vit import VisionTransformer
from src.dataset import build_imagenet_dataset
from src.merge_utils import MergeNet
from logger import logger, setup_logger


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, writer, rank, descript='Train'):
    model.train()
    total_loss, correct, total = 0, 0, 0

    if rank == 0:
        batch_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [{descript}]", leave=False)
    else:
        batch_bar = enumerate(loader)
    
    for step, (images, targets) in batch_bar:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        if rank == 0:
            batch_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })

    acc = 100. * correct / total
    if rank == 0 and writer is not None:
        writer.add_scalar("Train/Loss", total_loss / len(loader), epoch)
        writer.add_scalar("Train/Accuracy", acc, epoch)
    return total_loss / len(loader), acc


def validate(model, loader, criterion, device, epoch, writer, rank, merge=False, descript='Val'):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    if rank == 0:
        val_bar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch} [{descript}]", leave=False)
    else:
        val_bar = loader
    
    with torch.no_grad():
        for images, targets in val_bar:
            images, targets = images.to(device), targets.to(device)
            if merge == True:
                outputs = model.module.get_action(images)
            else:
                outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            if rank == 0:
                val_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })

    acc = 100. * correct / total
    if rank == 0 and writer is not None:
        writer.add_scalar("Val/Loss", total_loss / len(loader), epoch)
        writer.add_scalar("Val/Accuracy", acc, epoch)
    return total_loss / len(loader), acc


def save_checkpoint(model, epoch, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    torch.save(model.module.state_dict(), os.path.join(path, f"vit_epoch_{epoch}.pth"))

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='~/dataset/ImageNet')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--log-dir', type=str, default='results/vit')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')

    parser.add_argument('--opt', type=str, default='lawa')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--merge_number', type=int, default=5)
    parser.add_argument('--merge_k', type=int, default=2)
    parser.add_argument('--merge_epoch', type=int, default=1)
    parser.add_argument('--t', type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)

    log_dir = args.log_dir
    timestr = time.strftime("%y%m%d-%H%M%S")
    log_dir = log_dir + f'/{args.opt}_{args.seed}_{timestr}'
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    setup_logger(variant=vars(args), log_dir=log_dir)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    rank = dist.get_rank()

    # print(f"[RANK {dist.get_rank()}] Using GPU: {args.local_rank} / Total visible GPUs: {torch.cuda.device_count()}")

    if args.use_wandb and rank == 0:
        import wandb
        wandb.init(project="vit-imagenet", config=vars(args))

    model = VisionTransformer()
    model = model.to(device)
    model = DDP(model, device_ids=[args.local_rank])

    train_loader, val_loader, train_sampler = build_imagenet_dataset(
        args.data_path, args.batch_size, args.num_workers, distributed=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", disable=(rank != 0))

    state_dict_list = []
    for epoch in epoch_bar:
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, rank)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer, rank)

        state_dict_list.append(model.module.state_dict())
        
        if len(state_dict_list) == args.merge_number:
            ada_model = MergeNet(copy.deepcopy(model.module), state_dict_list, temperature=args.t, k=args.merge_k).to(device)
            ada_model.mask_logit.data[1::2] = 1.
            
            ada_model.get_model()
            infer_model = ada_model.infer_model.train()
            for p in infer_model.parameters():
                p.requires_grad = True
            model.module.load_state_dict(infer_model.state_dict())
            optimizer.state_dict()['state'].clear()  # 清空优化器的状态字典
            state_dict_list = []
            del ada_model
            gc.collect()
            torch.cuda.empty_cache()

        if rank == 0:
            tqdm.write(f"[Epoch {epoch}] "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if args.use_wandb:
                wandb.log({
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "epoch": epoch
                })

            # if epoch % 1 == 0:
            #     save_checkpoint(model, epoch, log_dir)

    if rank == 0:
        writer.close()
        if args.use_wandb:
            wandb.finish()


if __name__ == '__main__':
    main()