import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SingleAttentionHead(config.head_size) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.num_heads * config.head_size, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        self_attention = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(self_attention))

class SingleAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        # These can all be combined into one linear layer
        self.key = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.query = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.head_size = config.head_size

    
    def forward(self, x):
        B, T, C = x.shape # (batch, time, context_length)

        k = self.key(x)
        q = self.query(x)

        w = q @ k.transpose(-2, -1) 
        w *= self.head_size ** (-0.5)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (batch, time, time)
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        v = self.value(x)
        return w @ v

class CausalSelfAttention(nn.Module): # This acts a decoder
    def __init__(self, config):
        super().__init__()

        assert config.n_embed % config.n_head == 0, "Heads need to divide into embed dimension"

        self.kqv_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=False)
        self.proj = nn.Linear(config.n_embed, config.n_embed, bias=False)

        # Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout

        # Should support flash attention? TODO: read abou this
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() # (batch, seq_length, n_embed)

        # These are going to have a head dimension now vs the single head approach
        q, k, v = self.kqv_attn(x).split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, n_heads, seq_length, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, n_heads, seq_length, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, n_heads, seq_length, head_size)

        # This is where flash attention would go, pretty much all same as single head approach
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v # (batch, n_heads, seq_length, seq_length) x (batch, n_heads, seq_length, head_size) -> (batch, n_heads, seq_length, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # get us back to original shape

        return self.resid_dropout(self.proj(out))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed * 4), # inner space is higher dimension per paper
            nn.ReLU(),
            nn.Linear(config.n_embed * 4, config.n_embed),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # Basically groups attention with feedforward computation

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embed)
        self.sa = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.ffwd = MLP(config)

    def forward(self, x):
        # Add residual connections to stabilize training
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed) # final layer norm
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx/targets (B, T)
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x) # does this per token, ones communcation has happened from attention
        x = self.ln_f (x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, self.vocab_size)

            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (Batch, Time) of indicies cur context
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.block_size:]
            logits, loss = self(idx_crop)

            # Last time step
            logits = logits[:, -1, :] # (Batch, Channels)
            probs = F.softmax(logits, dim=-1) # (Batch, Channels)
            idx_next = torch.multinomial(probs, num_samples=1) #(Batch, 1)
            idx = torch.cat((idx, idx_next), dim=1) #(Batch, Time + 1)
        return idx # (Batch, Time + max_new_tokens)