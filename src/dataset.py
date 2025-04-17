import torchvision.transforms as transforms
import torchvision.datasets as datasets1
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
# from torchtext.datasets import IMDB
# from torchtext.data.utils import get_tokenizer
# from collections import Counter
# from torchtext.vocab import vocab
# from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from datasets import load_dataset
import numpy as np
import torch


# 数据加载
def get_data_loader(bs=256):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    train_dataset = datasets1.CIFAR100(root="/ailab/user/hushengchao/GDT/mask/data/", train=True, download=False, transform=transform_train)
    test_dataset = datasets1.CIFAR100(root="/ailab/user/hushengchao/GDT/mask/data/", train=False, download=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader

def get_dataloader_cifar10(args):
    data = np.load(args.data_path, allow_pickle=True).item()
    train_dataset = TensorDataset(
        torch.tensor(data[f'train_x{args.data}'], dtype=torch.float32),
        torch.tensor(data[f'train_y{args.data}'])
    )
    if args.eval:
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.bs,
            sampler=RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset))
        )

    val_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False)

    test_dataset = TensorDataset(
        torch.tensor(data['test_x'], dtype=torch.float32), torch.tensor(data['test_y'])
    )
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
    return train_loader, val_loader, test_loader

def get_agdata_loader(bs=32):
    # 加载 AG News 数据集
    dataset = load_dataset("ag_news")

    # 分词器
    tokenizer = get_tokenizer("basic_english")

    # 构建词汇表
    counter = Counter()
    for item in dataset["train"]["text"]:
        counter.update(tokenizer(item))

    vocab = vocab(counter, min_freq=5, specials=("<unk>", "<pad>"))
    vocab.set_default_index(vocab["<unk>"])

    # 文本和标签的预处理函数
    def text_pipeline(text):
        return [vocab[token] for token in tokenizer(text)]

    def label_pipeline(label):
        return label  # 标签本身是整数（0-3），无需处理

    # 数据集处理函数
    def collate_batch(batch):
        texts, labels = zip(*batch)
        text_indices = [torch.tensor(text_pipeline(text)) for text in texts]
        labels = torch.tensor([label_pipeline(label) for label in labels])
        text_indices = pad_sequence(text_indices, batch_first=True, padding_value=vocab["<pad>"])
        return text_indices, labels

    # DataLoader
    train_loader = DataLoader(
        list(zip(dataset["train"]["text"], dataset["train"]["label"])),
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_batch
    )

    test_loader = DataLoader(
        list(zip(dataset["test"]["text"], dataset["test"]["label"])),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_batch
    )

    return train_loader, test_loader

def build_imagenet_dataset(data_path, batch_size, workers, distributed):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = datasets1.ImageFolder(
        root=f"{data_path}/train",
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    val_dataset = datasets1.ImageFolder(
        root=f"{data_path}/val",
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )

    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )

    return train_loader, val_loader, train_sampler