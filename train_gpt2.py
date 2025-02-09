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

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

class Block(nn.Module): # class initialized for hidden layers
  def __init__(self, config):
    super().__init__()
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

def forward(self, idx): # token indices
    B, T = idx.size() # idx should be of shape (B, T)
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # loop through 0 to T --> create pos indices, making sure they are on the same device
    pos_emb = self.transformer.wpe(pos)
    tok_emb = self.transformer.wte(idx)
    x = tok_emb + pos_emb # broadcasting hidden in this
    for block in self.transformer.h: # forward blocks of the transformer
        x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)
    return logits # calculate logits for the next BxT token = what is the BxT + 1 logit

@classmethod    # returns the GPT object given the object type
def from_pretrained(cls, model_type, override_args=None):
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    override_args = override_args or {} # default to empty dict
    # only dropout can be overridden see more notes below
    assert all(k == 'dropout' for k in override_args)
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    # n_layer, n_head and n_embd are determined from model_type
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    print("forcing vocab_size=50257, block_size=1024, bias=True")
    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    config_args['bias'] = True # always True for GPT model checkpoints
    # we can override the dropout rate, if desired
    if 'dropout' in override_args:
        print(f"overriding dropout rate to {override_args['dropout']}")
        config_args['dropout'] = override_args['dropout']
    # create a from-scratch initialized minGPT model
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # copy while ensuring all of the parameters are aligned and match in names and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return model






# -----------------------------------QUICK TEST -- GENERATE 5 SENTENCES----------------------------------- #
import tiktoken

num_return_seq = 5
max_length = 30
model = GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda')


enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I am a Language Model, ")
tokens = torch.tensor(tokens, dtype=torch.long) # creating a torch tensor out of our tokens
tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1) # repeating it 5 times
x = tokens.to('cuda')

# generating next logits, x = (B, T) where B = 5, and T=8
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length: #max_length = 30
  with torch.no_grad():
    # code block to keep generating next token until we reach 30
    logits = model(x) # (B, T, vocab_size)
    logits = logits[:, -1, :] # (B, vocab_size)

    # get probabilities for the next token contenders
    probs = F.softmax(logits, dim=-1) # (B, vocab_size)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # 50 by default --> only keep top 50 probability, others 0
    ix = torch.multinomial(topk_probs, 1) # (B, 1) -- select the token
    xcol = torch.gather(topk_indices, -1, ix)
    x = torch.cat((x, xcol), dim=1) 

# finally we get x = (5, 30)
# printing out x
for i in range(num_return_seq):
  tokens = x[i, :max_length].tolist()
  decoded = enc.decode(tokens)
  print(f"> {decoded}")