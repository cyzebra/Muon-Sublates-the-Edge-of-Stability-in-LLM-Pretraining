import json
import os
from pathlib import Path
import numpy as np
import torch


def prepare(c):
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    from datasets import load_dataset
    import tiktoken
    root = Path(c['data_dir'])
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'manifest.json'
    if manifest.exists():
        return
    encoder = tiktoken.get_encoding('gpt2')
    args = dict(path=c['dataset_name'], split=c['dataset_split'], streaming=True)
    if c.get('dataset_config'):
        args['name'] = c['dataset_config']
    stream = iter(load_dataset(**args))
    sizes = {}
    for split in ('val', 'train'):
        target = c[f'prepare_{split}_tokens']
        array = np.memmap(root / f'{split}.bin', dtype='<u2', mode='w+', shape=(target,))
        written = 0
        while written < target:
            ids = encoder.encode_ordinary(next(stream)[c['text_field']]) + [encoder.eot_token]
            amount = min(len(ids), target - written)
            array[written:written + amount] = ids[:amount]
            written += amount
        array.flush()
        sizes[split] = target
    manifest.write_text(json.dumps(dict(train_file='train.bin', val_file='val.bin', dtype='<u2', train_tokens=sizes['train'], val_tokens=sizes['val'])))


class Tokens:
    def __init__(self, c):
        root = Path(c['data_dir'])
        if c.get('train_file'):
            tr = Path(c['train_file'])
            va = Path(c['val_file'])
            dtype = c['token_dtype']
            train = np.load(tr, mmap_mode='r') if tr.suffix == '.npy' else np.memmap(tr, dtype=dtype, mode='r')
            val = np.load(va, mmap_mode='r') if va.suffix == '.npy' else np.memmap(va, dtype=dtype, mode='r')
            self.train = train[c.get('train_start') or 0:c.get('train_stop')]
            self.val = val[c.get('val_start') or 0:c.get('val_stop')]
        else:
            m = json.loads((root / 'manifest.json').read_text())
            self.train = np.memmap(root / m['train_file'], dtype=m['dtype'], mode='r')
            self.val = np.memmap(root / m['val_file'], dtype=m['dtype'], mode='r')
    def starts(self, split, count, length, seed):
        return np.random.default_rng(seed).integers(0, len(getattr(self, split)) - length, size=count)
    def batches(self, split, starts, length, micro, device, vocab):
        data = getattr(self, split)
        for i in range(0, len(starts), micro):
            indices = starts[i:i + micro]
            z = np.stack([np.asarray(data[int(s):int(s) + length + 1], dtype=np.int64) for s in indices])
            yield torch.from_numpy(z[:, :-1].copy()).to(device), torch.from_numpy(z[:, 1:].copy()).to(device), len(indices) / len(starts)
