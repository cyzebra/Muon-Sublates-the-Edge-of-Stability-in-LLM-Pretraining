# Muon Training Experiments

Training and diagnostics for 22M, 130M, and 1B language models, plus CIFAR-10 MLP/CNN and Tiny Shakespeare Transformer experiments.

## Quick start

Install a PyTorch build compatible with your hardware, then:

```bash
python 22M/train.py --prepare
python 22M/train.py
python 130M/train.py --prepare
python 130M/train.py
python 1B/train.py --prepare
python 1B/train.py
python "neural network/train.py"
```

Each script loads its adjacent `config.json`; use `--config /absolute/path/config.json` to override it. Relative paths resolve inside the experiment directory. Language-model preparation streams FineWeb with GPT-2 tokenization; small-network datasets download automatically unless local sources are configured. Full runs require substantial compute and storage. Training uses one device.

## Experiments

| Setting | 22M | 130M | 1B |
| --- | --- | --- | --- |
| Effective batch | 16 | 32 | 512 |
| Sequence length | 4096 | 4096 | 4096 |
| Muon learning rate | 0.04 | 0.02 | 0.04 |
| Auxiliary AdamW learning rate | 0.001 | 0.001 | 0.001 |
| Steps / token budget | 8000 steps | 40 tokens/parameter | 20 tokens/parameter |

Language models use momentum-free NS-5 matrix updates with separate AdamW updates for embeddings and normalization parameters. Small networks use SVD-polar updates; configured batch sizes include full batch.

## Diagnostics

Language models write these files inside `output_dir`:

- `metrics.jsonl`: training loss, nuclear norm, paired reference-loss changes for Muon-only and full Muon+AdamW updates, realized-update `actual_muon_s/rho/edge`, and `R_muon/R_full`.
- `validation.jsonl`: validation loss and perplexity.
- `matrix_cpol.jsonl`: global, layer, module-type, and matrix cosine; secant quantities and residuals.
- `probes.jsonl`: conditional `s`, `rho`, `edge=2*rho/eta`, loss changes, confidence intervals, and finite-step identity residuals.
- `mc_samples.jsonl` and `block_coherence.jsonl`: individual probe draws and grouped alignment.

Global cosine uses a joint Frobenius inner product, not an average of matrix cosines. Sampled pairs always compare consecutive updates (lag 1); their `step` identifies the first update.

Probes use a fixed 32-sequence training reference, FP32 evaluation, NS-5 directions, and exact SVD nuclear norms. The reference approximates the population objective; NS-5 remains approximate polar. Muon-only `s/rho` exclude AdamW. Single realized-update diagnostics and conditional Monte Carlo averages are recorded separately.

Defaults from the supplied language-model reference are probes every 100 steps, Monte Carlo every 500 (8 draws; batches 1/8/16/32 plus training batch), cosine pairs every 5, and validation every 500. The actual 22M training-batch probe uses 16 sequences. The same configurable diagnostic defaults are supplied for 1B; they are not a claim about its historical diagnostic configuration.

Small networks write fixed-objective loss and global lag-1 cosine in `metrics.jsonl`, with conditional `s/rho/edge` and samples in `probes.jsonl`. The objective uses the full finite training set. Defaults are loss every step and probes every 25 steps with 8 draws; full batch uses one deterministic draw. Here cosine `step` identifies the second update.

## Checkpoints

`latest.pt` saves model/optimizer state and pending cosine history. Restarting resumes training and trims logs to the checkpoint. Frequencies and probe sizes are configurable; diagnostics add evaluation, SVD, and tensor-storage costs. No plots or reports are generated. Preserve configurations, data, and environment versions with results.
