import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from config import ARCH
from data import Tokens, prepare
from model import LM


def ns5(x, c):
    z = x.to(getattr(torch, c['ns_dtype']))
    transposed = z.shape[0] > z.shape[1]
    if transposed:
        z = z.mT
    z = z / (z.norm() + c['ns_eps'])
    for _ in range(5):
        a = z @ z.mT
        z = 3.4445 * z + (-4.7750 * a + 2.0315 * (a @ a)) @ z
    return (z.mT if transposed else z).to(x.dtype)


def rate(c, step, total):
    if c['schedule'] == 'constant':
        return 1.0
    warmup = c.get('warmup_steps') or int(total * c['warmup_fraction'])
    decay = c.get('decay_steps') or int(total * c['decay_fraction'])
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if step < total - decay:
        return 1.0
    progress = (step - (total - decay)) / max(1, decay)
    return 1 - (1 - c['final_lr_fraction']) * progress


def save(path, model, optimizer, step, seed):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), seed=seed)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def train(c):
    preset = c['models'][0]
    seed = c['seeds'][0]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Tokens(c)
    model = LM(preset, c).to(device)
    matrix, auxiliary = model.groups()
    batch = c['train_batch_sizes_by_model'][preset]
    total = c.get('train_steps') or math.ceil(c['tokens_per_parameter'] * sum(p.numel() for p in model.parameters()) / (batch * c['sequence_length']))
    if c['optimizer'] == 'all_adamw':
        groups = [dict(params=list(matrix.values()), weight_decay=c['adam_matrix_weight_decay']),
                  dict(params=list(auxiliary.values()), weight_decay=c['adam_weight_decay'])]
    else:
        groups = [dict(params=list(auxiliary.values()), weight_decay=c['adam_weight_decay'])]
    optimizer = torch.optim.AdamW(groups, lr=c['adam_lrs'][preset], betas=tuple(c['adam_betas']), eps=c['adam_eps'])
    output = Path(c['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    latest = output / 'latest.pt'
    start = 0
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        save(latest, model, optimizer, 0, seed)
    for step in range(start, total):
        factor = rate(c, step, total)
        for group in optimizer.param_groups:
            group['lr'] = c['adam_lrs'][preset] * factor
        model.zero_grad(set_to_none=True)
        starts = data.starts('train', batch, c['sequence_length'], np.random.SeedSequence([seed, 1729, step]))
        for x, y, weight in data.batches('train', starts, c['sequence_length'], c['micro_batch_size'], device, c['vocab_size']):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda' and c['train_precision'] == 'bf16'):
                loss = model(x, y) * weight
            loss.backward()
        if c.get('grad_clip_norm'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip_norm'])
        if c['optimizer'] == 'muon':
            with torch.no_grad():
                for p in matrix.values():
                    if p.grad is not None:
                        p.add_(ns5(p.grad, c), alpha=-c['muon_lrs'][preset] * factor)
        optimizer.step()
        if (step + 1) % c['checkpoint_every'] == 0 or step + 1 == total:
            save(latest, model, optimizer, step + 1, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    c = json.loads(Path(args.config).read_text())
    if args.prepare:
        prepare(c)
    else:
        train(c)


if __name__ == '__main__':
    main()
