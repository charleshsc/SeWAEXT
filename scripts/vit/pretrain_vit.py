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
from src.swa import update_bn_custom

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


def save_checkpoint(model, optimizer, scheduler, epoch, log_dir):
    if dist.get_rank() == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'log_dir': log_dir,  # 保存 TensorBoard 日志路径
        }, log_dir + '/checkpoint.pth')

def load_checkpoint(model, optimizer, scheduler, path):
    path = path + '/checkpoint.pth'
    checkpoint = torch.load(path, map_location="cpu")

    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    log_dir = checkpoint.get("log_dir")

    return start_epoch, log_dir

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

    parser.add_argument('--opt', type=str, default='merge')
    parser.add_argument('--resume', type=str, default='/ailab/user/hushengchao/GDT/opt/results/vit/merge_123_250424-095719')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--merge_number', type=int, default=4)
    parser.add_argument('--merge_k', type=int, default=2)
    parser.add_argument('--merge_epoch', type=int, default=1)
    parser.add_argument('--t', type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    rank = dist.get_rank()

    # print(f"[RANK {dist.get_rank()}] Using GPU: {args.local_rank} / Total visible GPUs: {torch.cuda.device_count()}")

    model = VisionTransformer()
    model = model.to(device)
    model = DDP(model, device_ids=[args.local_rank])

    train_loader, val_loader, train_sampler = build_imagenet_dataset(
        args.data_path, args.batch_size, args.num_workers, distributed=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.resume is None:
        log_dir = args.log_dir
        timestr = time.strftime("%y%m%d-%H%M%S")
        log_dir = log_dir + f'/{args.opt}_{args.seed}_{timestr}'
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        setup_logger(variant=vars(args), log_dir=log_dir)
        start_epoch = 1
    else:
        start_epoch, log_dir = load_checkpoint(model, optimizer, None, args.resume)
        setup_logger(variant=vars(args), log_dir=log_dir)

    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    epoch_bar = tqdm(range(start_epoch, args.epochs + 1), desc="Epochs", disable=(rank != 0))

    state_dict_list = []
    for epoch in epoch_bar:
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, rank)

        if 'merge' in args.opt:
            if epoch % 5 != 1:
                state_dict_list.append(model.module.state_dict())
            
            if len(state_dict_list) == args.merge_number:
                eval_logits = []
                val_accs = []
                ada_model = MergeNet(copy.deepcopy(model.module), state_dict_list, temperature=args.t, k=args.merge_k).to(device)
                ada_model = DDP(ada_model, device_ids=[args.local_rank])
                ada_optimizer = torch.optim.AdamW(ada_model.module.collect_trainable_params(), lr=args.lr)

                train_loader_, val_loader_, train_sampler_ = build_imagenet_dataset(
                    args.data_path, 64, args.num_workers, distributed=True
                )
                for merge_epoch in range(args.merge_epoch):
                    if train_sampler_:
                        train_sampler_.set_epoch(merge_epoch)

                    train_loss_, train_acc_ = train_one_epoch(ada_model, train_loader_, ada_optimizer, criterion, device, epoch, None, rank, 'Merge')
                    logit = ada_model.module.get_model(rank=rank)
                    # val_loss_, val_acc_ = validate(ada_model, val_loader_, criterion, device, epoch, None, rank, True, 'Merge Val')
                    eval_logits.append(ada_model.module.mask_logit.detach().cpu().numpy())   
                    # val_accs.append(val_acc_)
                
                ada_model.module.get_model()
                infer_model = ada_model.module.infer_model.train()
                for p in infer_model.parameters():
                    p.requires_grad = True
                model.module.load_state_dict(infer_model.state_dict())
                # optimizer.state_dict()['state'].clear()  # 清空优化器的状态字典
                state_dict_list = []
                del ada_model
                del ada_optimizer
                gc.collect()
                torch.cuda.empty_cache()

                if rank == 0:
                    logger.log(f'epoch {epoch}')
                    logger.log(str(eval_logits))
                    # logger.log(str(val_accs))
            
                update_bn_custom(train_loader, model.module)
        
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer, rank)

        if rank == 0:
            tqdm.write(f"[Epoch {epoch}] "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            logger.log(f"[Epoch {epoch}] "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # if epoch % 1 == 0:
            #     save_checkpoint(model, epoch, log_dir)
        
            save_checkpoint(model, optimizer, None, epoch, log_dir)

    if rank == 0:
        writer.close()


if __name__ == '__main__':
    main()