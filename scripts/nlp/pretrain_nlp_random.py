import time
import argparse
import random
import numpy as np
from pathlib import Path
import os
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

from src.dataset import get_data_loader
from src.model import TransformerClassifier
from src.merge_utils import MergeNet
from src.swa import update_bn_custom

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def test(model, test_loader, device, name):
    # 测试模型
    model.eval()
    total_correct, total_samples = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts)  # [B, num_classes]
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"{name} Accuracy: {accuracy:.4f}")
    return accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='~/dataset/ImageNet')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--log_dir', type=str, default='results/')
parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')

parser.add_argument('--embed_size', type=int, default=128)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--max_len', type=int, default=512)

parser.add_argument('--opt', type=str, default='random')
parser.add_argument('--save_interval', type=int, default=1)
parser.add_argument('--merge_number', type=int, default=50)
parser.add_argument('--merge_k', type=int, default=10)
parser.add_argument('--merge_epoch', type=int, default=1)
parser.add_argument('--t', type=float, default=1.0)
args = parser.parse_args()
set_seed(args.seed)

timestr = time.strftime("%y%m%d-%H%M%S")
results_dir = f'{args.log_dir}/NLP/{args.opt}-{args.seed}-{timestr}'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 加载 AG News 数据集
dataset = load_dataset("ag_news")
tokenizer = get_tokenizer("basic_english")
counter = Counter()
for item in dataset["train"]["text"]:
    counter.update(tokenizer(item))
vocab = vocab(counter, min_freq=5, specials=("<unk>", "<pad>"))
vocab.set_default_index(vocab["<unk>"])
vocab_size = len(vocab)

# 数据集处理函数
def collate_batch(batch):

    # 文本和标签的预处理函数
    def text_pipeline(text):
        return [vocab[token] for token in tokenizer(text)]

    def label_pipeline(label):
        return label  # 标签本身是整数（0-3），无需处理
    
    texts, labels = zip(*batch)
    text_indices = [torch.tensor(text_pipeline(text)) for text in texts]
    labels = torch.tensor([label_pipeline(label) for label in labels])
    text_indices = pad_sequence(text_indices, batch_first=True, padding_value=vocab["<pad>"])
    return text_indices, labels

# DataLoader
train_loader = DataLoader(
    list(zip(dataset["train"]["text"], dataset["train"]["label"])),
    batch_size=32,
    shuffle=True,
    collate_fn=collate_batch
)
val_loader = DataLoader(
    list(zip(dataset["train"]["text"], dataset["train"]["label"])),
    batch_size=32,
    shuffle=True,
    collate_fn=collate_batch
)
test_loader = DataLoader(
    list(zip(dataset["test"]["text"], dataset["test"]["label"])),
    batch_size=32,
    shuffle=False,
    collate_fn=collate_batch
)

# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerClassifier(vocab_size, args.embed_size, args.num_heads, args.hidden_dim, args.num_layers, args.num_classes, args.max_len).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练循环
global_iter = 0
train_acc = []
val_acc = []
state_dict_list = []
for epoch in range(args.epochs):
    model.train()
    total_loss, total_correct = 0, 0

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()

        if global_iter % args.save_interval == 0:
            state_dict_list.append(model.state_dict())
        
        if len(state_dict_list) == args.merge_number:
            ada_model = MergeNet(copy.deepcopy(model), state_dict_list, temperature=args.t, k=args.merge_k).to(device)
            random_numbers = random.sample(range(args.merge_number), args.merge_k)
            for i in random_numbers:
                ada_model.mask_logit.data[i] = 1.
            
            ada_model.get_model()
            infer_model = ada_model.infer_model

            model.load_state_dict(infer_model.state_dict())
            optimizer.state_dict()['state'].clear()  # 清空优化器的状态字典
            state_dict_list = []

        global_iter += 1
        if global_iter % 100 == 0:
            update_bn_custom(train_loader, model)
            train_acc.append(test(model, train_loader, device, 'Train'))
            val_acc.append(test(model, test_loader, device, 'Test'))

with open(os.path.join(results_dir, 'train_acc.npy'), 'wb') as f:
    np.save(f, np.array(train_acc))
with open(os.path.join(results_dir, 'val_acc.npy'), 'wb') as f:
    np.save(f, np.array(val_acc))

plt.figure()
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.legend()
plt.savefig(os.path.join(results_dir, 'acc.png'))
plt.close()