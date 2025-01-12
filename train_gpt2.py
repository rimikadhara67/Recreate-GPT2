from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
  # pyTorhc optimized version of Multi_head Attention
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0

    self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd) # k, q, v projectstion for all heads, in a batch
    self.c_proj = nn.Linear(config.n_embd, config.n_embd) # final output projections
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    # at this point, we have tokens lined up in a sequence (1024 tokens) -- each token emits three vectors = k, q, v
    B, T, C = x.size()
    qkv = self.c_attn(x)     # -- queries and keys have to find relationships amongst each other -- done through the attention block
    q, k, v = qkv.split(self.n_embd, dim=2) # -- splitting into the three  vectors
    # making the num_heads into a batched_dimension -- to treat B and nh in batches to parallelize
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt((k.size(-1))))
    # -- masked attention = autoregressive, making tokens attend to what's before them instead of in the future
    # -- ensures they don't learn to predict the future by already knowing the future. 
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) 
    att = F.softmax(att, dim=-1) #-- normalizes the attention
    y = att @ v # attention matmul with values = weighted sum of the values of the tokens that we found interesting -- which tokens does this token attend to
    y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble -- concat
    y = self.c_proj(y)
    return y

class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) # linear layer
    self.gelu = nn.GELU(approximate='tanh') # basically rely without a flat tail at 0 -- slightly smoother relu
    # another reason -- dead relu neuron problem -- 0 has no change or development of network
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # linear layer


class Block(nn.Module): # class initialized for hidden layers
  def __init_(self, config):
    self.ln_1 = nn.LayerNorm(config.n_embd) # -- layer normalization
    self.attn = CausalSelfAttention(config) # -- self attention layer
    self.ln_2 = nn.LayerNorm(config.n_embd) # -- layer norm 2
    self.mlp = MLP(config)                  # -- feed forward network

  def forward(self, x): # a change from AIAYN paper
    x = x + self.attn(self.ln_1(x)) # x goes to layer norm, then attention layer -- weighted sum function -- reduce()
    x = x + self.mlp(self.ln_2(x))  # then, x goes to second layer norm, and then feed forward network -- no information collected -- map()
    # ?? residual stream
    return x
  

@dataclass
class GPTConfig:
  block_size: int = 1024 #max sequence lengths
  vocab_size: int = 50257   # num_tokens
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768         # dim embeddings   


class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    # skeleton NN Module -- we are trying to recreate the state_dict above
    # nn.Embedding = fancy wrapper module around single arrays of number -- allows ot access elements
    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd), #weights of token embd -- output embedding in the transformer arhchitecture
        wpe = nn.Embedding(config.block_size, config.n_embd), #weights of postiion embd -- PE is postional encodings 
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden -- we have 12 of those -- all the blocks of Attention, Add/norm, FeedForward
        ln_f = nn.LayerNorm(config.n_embd), # final layer norm -- new thing added to GPT2
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # final classifer -- Linear layer at the end before softmax