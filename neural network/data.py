import pickle
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


class Objective:
    def __init__(self, name, x, y, meta):
        self.name, self.x, self.y, self.meta = name, x, y, meta
        self.n = len(x)
    def batches(self, indices, micro, device, dtype):
        n = self.n if indices is None else len(indices)
        for start in range(0, n, micro):
            ix = slice(start, start + micro) if indices is None else indices[start:start + micro]
            x, y = self.x[ix].to(device), self.y[ix].to(device)
            if self.name != 'transformer':
                x, y = x.to(dtype), y.to(dtype)
            yield x, y
    def denominator(self, indices):
        n = self.n if indices is None else len(indices)
        return n * self.y.shape[1] if self.name == 'transformer' else n
    def loss_sum(self, logits, y):
        if self.name != 'transformer':
            return .5 * (logits - y).double().square().sum()
        return F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='sum').double()


def cifar_folder(c):
    root = Path(c['CIFAR_SOURCE']) if c['CIFAR_SOURCE'] else Path('data/cifar-10-batches-py')
    if (root / 'data_batch_1').exists():
        return root
    if (root / 'cifar-10-batches-py/data_batch_1').exists():
        return root / 'cifar-10-batches-py'
    archive = root if root.is_file() else Path('data/cifar-10-python.tar.gz')
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', archive)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if member.isfile() and not path.is_absolute() and '..' not in path.parts and path.parts[0] == 'cifar-10-batches-py':
                dest = Path('data') / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dest.open('wb') as dst:
                    dst.write(src.read())
    return Path('data/cifar-10-batches-py')


def load_objective(name, c):
    if name in ('mlp', 'cnn'):
        folder = cifar_folder(c)
        parts, labels = [], []
        for i in range(1, 6):
            with (folder / f'data_batch_{i}').open('rb') as f:
                item = pickle.load(f, encoding='bytes')
            parts.append(item[b'data'])
            labels.extend(item[b'labels'])
        x = torch.from_numpy(np.concatenate(parts)).float().div_(255).reshape(-1, 3, 32, 32)
        x = (x - x.mean(0, keepdim=True)) / x.std(0, unbiased=False, keepdim=True).clamp_min(1e-12)
        y = F.one_hot(torch.tensor(labels), 10).float()
        indices = torch.randperm(len(x), generator=torch.Generator().manual_seed(c['DATA_SEED']))[:c['CIFAR_N']]
        return Objective(name, x[indices], y[indices], {})
    path = Path(c['TEXT_SOURCE']) if c['TEXT_SOURCE'] else Path('data/tinyshakespeare.txt')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', path)
    content = path.read_text(encoding='utf-8').replace('\r\n', '\n').replace('\r', '\n')
    chars = sorted(set(content))
    table = {character: i for i, character in enumerate(chars)}
    tokens = torch.tensor([table[character] for character in content[:int(.9 * len(content))]], dtype=torch.long)
    block = c['BLOCK_SIZE']
    starts = np.sort(np.random.default_rng(c['DATA_SEED']).choice(len(tokens) - block, c['TEXT_N'], replace=False))
    x = torch.stack([tokens[s:s + block] for s in starts])
    y = torch.stack([tokens[s + 1:s + block + 1] for s in starts])
    return Objective(name, x, y, dict(vocab_size=len(chars), block_size=block))
