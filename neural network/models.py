import math
import torch
from torch import nn
from torch.nn import functional as F

def seeded(seed):
    torch.manual_seed(seed)

class CIFARMLP(nn.Module):

    def __init__(self, widths=(128, 128), use_bias=True):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, widths[0], bias=use_bias)
        self.fc2 = nn.Linear(widths[0], widths[1], bias=use_bias)
        self.fc3 = nn.Linear(widths[1], 10, bias=use_bias)

    def forward(self, x):
        x = x.flatten(1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

class CIFARCNN(nn.Module):

    def __init__(self, use_bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=use_bias)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=use_bias)
        self.fc1 = nn.Linear(64 * 8 * 8, 128, bias=use_bias)
        self.fc2 = nn.Linear(128, 10, bias=use_bias)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2)
        x = torch.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        x = x.flatten(1)
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, block_size, use_bias=True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.projection = nn.Linear(d_model, d_model, bias=use_bias)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer('causal_mask', mask, persistent=False)

    def forward(self, x):
        b, t, c = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:t, :t]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.projection(out)

class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, block_size, mlp_multiplier=4, use_bias=True):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, block_size, use_bias)
        self.norm_2 = nn.LayerNorm(d_model)
        hidden = mlp_multiplier * d_model
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden, bias=use_bias), nn.GELU(), nn.Linear(hidden, d_model, bias=use_bias))

    def forward(self, x):
        x = x + self.attention(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x

class TinyShakespeareTransformer(nn.Module):

    def __init__(self, vocab_size, block_size=64, d_model=64, n_heads=4, n_layers=2, mlp_multiplier=4, use_bias=True):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, block_size, mlp_multiplier, use_bias) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        b, t = token_ids.shape
        if t > self.block_size:
            raise ValueError()
        pos = torch.arange(t, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(pos)
        for block in self.blocks:
            x = block(x)
        return self.output_head(self.final_norm(x))

def build_model(name, obj, seed, device, dtype):
    seeded(seed)
    if name == 'mlp':
        model = CIFARMLP()
    elif name == 'cnn':
        model = CIFARCNN()
    else:
        model = TinyShakespeareTransformer(obj.meta['vocab_size'], block_size=obj.meta['block_size'])
    return model.to(device=device, dtype=dtype).eval()
