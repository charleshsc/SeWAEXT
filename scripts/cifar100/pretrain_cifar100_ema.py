import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import os
import gc
import math
import argparse
import copy
import torch
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from decimal import Decimal

from src.dataset import get_data_loader
from src.model import ResNet5, ResNet3, ResNet1
from src.merge_utils import MergeNet
from src.ema import EMA

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 训练函数
def train_step(device, epoch, model, train_loader, optimizer, criterion, val_loader, descript='Train', args=None, ema_model=None, merge_k=1):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    state_dict_list = []
    batch_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [{descript}]", leave=False)

    for batch_idx, (inputs, targets) in batch_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        batch_bar.set_postfix({
                'loss': f"{train_loss/(batch_idx+1):.3f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        ema_model.update(model)

        if (batch_idx+1) % merge_k == 0:
            ema_model.apply(model)
    
    ema_model.apply(model)
    return train_loss/(batch_idx+1)

# 测试函数
def test(device, model, loader, criterion, name='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    batch_bar = tqdm(enumerate(loader), total=len(loader), desc=f"[{name}]", leave=False)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            batch_bar.set_postfix({
                'loss': f"{test_loss/(batch_idx+1):.3f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
    
    return 100.*correct/total

def main():
    parser = argparse.ArgumentParser(description="Stability")
    parser.add_argument("--data_name", type=str, default="cifar100")
    parser.add_argument('--data', type=int, default=1,
                        choices=[1, 2],)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model", type=str, default="ResNet3")

    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--start_save", type=int, default=1)
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_max_iter", type=int, default=100000)
    parser.add_argument('--log_dir', type=str, default='results/')

    # optimizer
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr_decay", action='store_true')

    parser.add_argument('--opt', type=str, default='ema')
    parser.add_argument('--merge_number', type=int, default=50)
    parser.add_argument('--merge_k', type=int, default=10)
    parser.add_argument('--merge_epoch', type=int, default=1)
    parser.add_argument('--t', type=float, default=1.0)
    
    args = parser.parse_args()
    set_seed(args.seed)

    # 数据加载
    train_loader, val_loader, test_loader = get_data_loader(args.bs)

    # 训练配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "ResNet1" in args.model:
        model = ResNet1(num_classes=100).to(device)
    elif "ResNet3" in args.model:
        model = ResNet3(num_classes=100).to(device)
    elif "ResNet5" in args.model:
        model = ResNet5(num_classes=100).to(device)
    ema = EMA(model, decay=0.9)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    timestr = time.strftime("%y%m%d-%H%M%S")
    log_path = f'{args.log_dir}/Cifar100/{args.opt}-{args.model}-{args.seed}-{timestr}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    learning_rates_log = []
    train_acc = []
    val_acc = []
    train_losses = []

    # 主循环
    for epoch in trange(args.epochs, desc="Epochs",):
        train_loss = train_step(device, epoch, model, train_loader, optimizer, criterion, val_loader, 'Train', args, ema, args.merge_number)
        train_acc.append(test(device, model, train_loader, criterion, 'Train_ACC'))
        val_acc.append(test(device, model, test_loader, criterion, 'Test_ACC'))
        train_losses.append(train_loss)
        scheduler.step()
        learning_rates_log.append(scheduler.get_last_lr())

    with open(os.path.join(log_path, 'train_acc.npy'), 'wb') as f:
        np.save(f, np.array(train_acc))
    with open(os.path.join(log_path, 'val_acc.npy'), 'wb') as f:
        np.save(f, np.array(val_acc))
    with open(os.path.join(log_path, 'train_loss.npy'), 'wb') as f:
        np.save(f, np.array(train_losses))
    
    plt.figure()
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend()
    plt.savefig(os.path.join(log_path, 'acc.png'))
    plt.close()
    if scheduler is not None:
        with open(os.path.join(log_path, 'learning_rates.npy'), 'wb') as f:
            np.save(f, np.array(learning_rates_log))

if __name__ == "__main__":
    main()