import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from config import ARCH

class RMSNorm(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-05)).to(x.dtype) * self.weight

class Attention(nn.Module):

    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.dh = d // h
        if d % h or self.dh % 2:
            raise ValueError()
        for n in ['q', 'k', 'v', 'o']:
            setattr(self, n, nn.Linear(d, d, bias=False))
        self.register_buffer('freq', 1.0 / 10000 ** (torch.arange(0, self.dh, 2).float() / self.dh), persistent=False)

    def rope(self, x):
        t = torch.arange(x.shape[-2], device=x.device).float()
        a = t[:, None] * self.freq[None, :]
        co = a.cos().to(x.dtype)
        si = a.sin().to(x.dtype)
        e, o = (x[..., 0::2], x[..., 1::2])
        return torch.stack((e * co - o * si, e * si + o * co), dim=-1).flatten(-2)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = [getattr(self, n)(x).view(b, t, self.h, self.dh).transpose(1, 2) for n in ['q', 'k', 'v']]
        z = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, is_causal=True, dropout_p=0.0)
        return self.o(z.transpose(1, 2).contiguous().view(b, t, d))

class MLP(nn.Module):

    def __init__(self, d, f):
        super().__init__()
        self.gate = nn.Linear(d, f, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):

    def __init__(self, d, h, f):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, h)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, f)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class LM(nn.Module):

    def __init__(self, preset, c):
        super().__init__()
        a = ARCH[preset]
        self.c = c
        self.embed = nn.Embedding(c['vocab_size'], a['width'])
        self.blocks = nn.ModuleList([Block(a['width'], a['heads'], a['ffn']) for _ in range(a['layers'])])
        self.norm = RMSNorm(a['width'])
        self.apply(self.initialize)
        for b in self.blocks:
            for p in [b.attn.o.weight, b.mlp.down.weight]:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * a['layers']))

    @staticmethod
    def initialize(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, y):
        z = self.embed(x)
        for b in self.blocks:
            z = checkpoint(b, z, use_reentrant=False) if self.c['activation_checkpointing'] and torch.is_grad_enabled() else b(z)
        z = self.norm(z).flatten(0, 1)
        y = y.flatten()
        loss = z.new_zeros((), dtype=torch.float32)
        for start in range(0, len(y), self.c['loss_chunk_tokens']):
            h = z[start:start + self.c['loss_chunk_tokens']]
            target = y[start:start + self.c['loss_chunk_tokens']]

            def ce(h, w, target):
                return F.cross_entropy(F.linear(h, w).float(), target, reduction='sum')
            term = checkpoint(ce, h, self.embed.weight, target, use_reentrant=False) if self.c['activation_checkpointing'] and torch.is_grad_enabled() else ce(h, self.embed.weight, target)
            loss = loss + term
        return loss / len(y)

    def groups(self):
        mu = {}
        aux = {}
        for n, p in self.named_parameters():
            (mu if n.startswith('blocks.') and p.ndim == 2 else aux)[n] = p
        assert set(mu).isdisjoint(aux) and len(mu) + len(aux) == len(list(self.parameters()))
        return (mu, aux)
