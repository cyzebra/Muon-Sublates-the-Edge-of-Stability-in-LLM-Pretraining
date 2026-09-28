import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from data import load_objective
from models import build_model


def polar(gradients):
    result = []
    for g in gradients:
        matrix = g.reshape(g.shape[0], -1) if g.ndim >= 2 else g.reshape(-1, 1)
        if min(matrix.shape) == 1:
            norm = torch.linalg.vector_norm(matrix)
            direction = matrix / norm if norm > 0 else torch.zeros_like(matrix)
        else:
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
            threshold = max(matrix.shape) * torch.finfo(s.dtype).eps * s.max()
            mask = s > threshold
            direction = u[:, mask] @ vh[mask] if mask.any() else torch.zeros_like(matrix)
        result.append(direction.reshape_as(g))
    return result


def train_one(c, name, batch, eta, seed):
    objective = load_objective(name, c)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(name, objective, seed, device, getattr(torch, c['DTYPE']))
    parameters = [p for p in model.parameters() if p.requires_grad]
    folder = Path('runs') / name / f'b{batch}_lr{eta:g}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / 'latest.pt'
    start = 0
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        start = state['step']
    for step in range(start, c['STEPS']):
        if batch == 'full':
            indices = None
        else:
            maximum = max(int(b) for b in c['BATCHES'][name] if b != 'full' and int(b) < objective.n)
            indices = np.random.default_rng(np.random.SeedSequence([c['BATCH_SEED'], seed, step])).choice(objective.n, maximum, replace=False)[:int(batch)]
        gradients = [torch.zeros_like(p) for p in parameters]
        for x, y in objective.batches(indices, c['MICROBATCH'][name], device, parameters[0].dtype):
            loss = objective.loss_sum(model(x), y) / objective.denominator(indices)
            parts = torch.autograd.grad(loss, parameters)
            for accumulator, part in zip(gradients, parts):
                accumulator.add_(part.detach())
        direction = polar(gradients)
        with torch.no_grad():
            for parameter, update in zip(parameters, direction):
                parameter.add_(update, alpha=-eta)
        if (step + 1) % c['CKPT_EVERY'] == 0 or step + 1 == c['STEPS']:
            tmp = latest.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(), step=step + 1), tmp)
            os.replace(tmp, latest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    c = json.loads(Path(args.config).read_text())
    for name in c['MODELS']:
        for batch in c['RUN_BATCHES'][name]:
            for eta in c['LEARNING_RATES']:
                for seed in c['TRAIN_SEEDS']:
                    train_one(c, name, batch, eta, seed)


if __name__ == '__main__':
    main()
