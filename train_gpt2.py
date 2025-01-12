from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

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
  block_size: int = 256
  vocab_size: int = 65
  n_layer: int = 6
  n_head: int = 6
  n_embd: int = 384


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