# fl_framework/data/datasets.py
from typing import List, Any, Tuple
from torchvision import datasets, transforms
from .base_dataset import BaseFLDataset
import re
import torch
import csv
import os
import collections
import random
from torch.utils.data import random_split

class MNISTDataset(BaseFLDataset):

    """
    An adapter for the torchvision MNIST dataset to make it compatible
    with the BaseFLDataset interface.
    """
    num_classes: int = 10
    def __init__(self, data_dir: str = './data_cache', train: bool = True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(
            data_dir, train=train, download=True, transform=transform
        )

    @property
    def targets(self) -> List[int]:
        return self.dataset.targets.tolist()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return (index,) + self.dataset[index]

class CIFAR10Dataset(BaseFLDataset):
    """
    An adapter for the torchvision CIFAR-10 dataset to make it compatible
    with the BaseFLDataset interface.
    """
    num_classes: int = 10
    def __init__(self, data_dir: str = './data_cache', train: bool = True):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]), 
            transforms.RandomErasing(),
        ])
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]), 
        ])
        self.dataset = datasets.CIFAR10(
            data_dir, train=train, download=True, transform=transform_train if train else transform
        )

    @property
    def targets(self) -> List[int]:
        return self.dataset.targets

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        out  = (index,) + self.dataset[index]
        return out

class Sentiment140Dataset(BaseFLDataset):
    """
    Sentiment140 dataset for sentiment analysis.
    Labels: 0 (negative), 1 (positive).
    """
    _num_classes: int = 2
    def __init__(self, data_file: str, train: bool = True, max_samples: int = None, vocab_file: str = None, max_len: int = 128, tokenizer_type: str = 'simple'):
        self.data_file = data_file
        self.max_len = max_len
        self.tokenizer_type = tokenizer_type
        self.data = []
        self.labels = []
        
        # Handle vocab file path (only for simple tokenizer)
        if vocab_file is None:
            data_dir = os.path.dirname(data_file)
            self.vocab_file = os.path.join(data_dir, 'vocab.pt')
        else:
            self.vocab_file = vocab_file
            
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
            
        # 1. Load Data
        with open(data_file, 'r', encoding='latin-1') as f:
            reader = csv.reader(f)
            # Read all rows to ensure we can shuffle and get balanced data if max_samples is used
            all_rows = list(reader)
            
        # Shuffle deterministically to ensure consistency across runs/clients if using same file
        if train:
            random.Random(42).shuffle(all_rows)
            
        if max_samples is not None:
            all_rows = all_rows[:max_samples]
            
        for row in all_rows:
            # row[0]: target, row[5]: text
            target = int(row[0])
            if target == 4:
                label = 1
            elif target == 0:
                label = 0
            else:
                continue # Skip neutral (target=2) if present
            
            self.labels.append(label)
            self.data.append(row[5]) # Raw text
        
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        # 2. Build or Load Vocab / Initialize Tokenizer
        if tokenizer_type != 'simple':
            raise ValueError('SentimentTransformer uses the simple tokenizer.')
        self.tokenizer = None
        self.vocab = self._load_or_build_vocab()
        self.pad_idx = self.vocab.get('<pad>', 0)
        self.unk_idx = self.vocab.get('<unk>', 1)

    def _simple_tokenizer(self, text: str) -> List[str]:
        # Simple whitespace tokenizer with lowercase
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        return text.lower().split()

    def _load_or_build_vocab(self) -> dict:
        if os.path.exists(self.vocab_file):
            print(f"Loading vocab from {self.vocab_file}")
            return torch.load(self.vocab_file)
        
        print(f"Building vocab from {self.data_file}...")
        counter = collections.Counter()
        for text in self.data:
            tokens = self._simple_tokenizer(text)
            counter.update(tokens)
            
        # Keep top 30,000 words
        most_common = counter.most_common(30000)
        vocab = {'<pad>': 0, '<unk>': 1}
        for word, _ in most_common:
            vocab[word] = len(vocab)
            
        print(f"Saving vocab to {self.vocab_file} (size: {len(vocab)})")
        torch.save(vocab, self.vocab_file)
        return vocab

    def text_to_indices(self, text: str) -> torch.Tensor:
        tokens = self._simple_tokenizer(text)
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens][:self.max_len]
        indices += [self.pad_idx] * (self.max_len-len(indices))
        return torch.tensor(indices, dtype=torch.long)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def targets(self) -> List[int]:
        return self.labels.tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        text = self.data[index]
        input_ids = self.text_to_indices(text)
        label = self.labels[index]
        return index, input_ids, label
