import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from config import ARCH

class RMSNorm(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-05)
        return y.to(x.dtype) * self.weight

class Attention(nn.Module):

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        if width % heads or self.head_dim % 2:
            raise ValueError()
        for name in ['q', 'k', 'v', 'o']:
            setattr(self, name, nn.Linear(width, width, bias=False))
        freq = 1.0 / 10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        self.register_buffer('freq', freq, persistent=False)

    def rope(self, x):
        positions = torch.arange(x.shape[-2], device=x.device).float()
        angles = positions[:, None] * self.freq[None, :]
        cos, sin = (angles.cos().to(x.dtype), angles.sin().to(x.dtype))
        even, odd = (x[..., 0::2], x[..., 1::2])
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = [getattr(self, name)(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2) for name in ['q', 'k', 'v']]
        z = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, is_causal=True, dropout_p=0.0)
        return self.o(z.transpose(1, 2).contiguous().view(batch, length, width))

class MLP(nn.Module):

    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):

    def __init__(self, width, heads, hidden):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = Attention(width, heads)
        self.norm2 = RMSNorm(width)
        self.mlp = MLP(width, hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class LM(nn.Module):

    def __init__(self, preset, config):
        super().__init__()
        arch = ARCH[preset]
        self.config = config
        self.embed = nn.Embedding(config['vocab_size'], arch['width'])
        self.blocks = nn.ModuleList([Block(arch['width'], arch['heads'], arch['ffn']) for _ in range(arch['layers'])])
        self.norm = RMSNorm(arch['width'])
        self.apply(self.initialize)
        for block in self.blocks:
            for parameter in [block.attn.o.weight, block.mlp.down.weight]:
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * arch['layers']))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, y):
        z = self.embed(x)
        for block in self.blocks:
            if self.config['activation_checkpointing'] and torch.is_grad_enabled():
                z = checkpoint(block, z, use_reentrant=False)
            else:
                z = block(z)
        z, y = (self.norm(z).flatten(0, 1), y.flatten())
        loss = z.new_zeros((), dtype=torch.float32)
        chunk = self.config['loss_chunk_tokens']
        for start in range(0, len(y), chunk):
            hidden, target = (z[start:start + chunk], y[start:start + chunk])

            def cross_entropy(h, weight, labels):
                return F.cross_entropy(F.linear(h, weight).float(), labels, reduction='sum')
            if self.config['activation_checkpointing'] and torch.is_grad_enabled():
                term = checkpoint(cross_entropy, hidden, self.embed.weight, target, use_reentrant=False)
            else:
                term = cross_entropy(hidden, self.embed.weight, target)
            loss = loss + term
        return loss / len(y)

    def groups(self):
        muon, auxiliary = ({}, {})
        for name, parameter in self.named_parameters():
            target = muon if name.startswith('blocks.') and parameter.ndim == 2 else auxiliary
            target[name] = parameter
        assert set(muon).isdisjoint(auxiliary)
        return (muon, auxiliary)
