import torch
import torch.nn as nn
import torch.nn.functional as F

from self_attn import SelfAttention
from self_attn import embedded_sentence

## transformers leverage multi-head attention
## in scaled dot-product attn the input sequence is converted to three component matrices (querys, keys, values)
## these three components together can be thought of as a single attention head
## the single head produces a single context vector wheres multiple context vectors are produces with multi-head
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v, num_heads): ## additional parameter num_heads
        super().__init__()
        ## d_in: dim of the input feature vector
        ## d_out_kq: dim of the query and key outputs
        ## d_out_v: dim of the value outputs
        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out_kq, d_out_v)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        ## note that the forward pass is applying self attn to the input independently for each head
        ## results are tehn concatenated along the final dimension
        return torch.cat([head(x) for head in self.heads], dim=-1)

# single attention head
torch.manual_seed(123)
d_in, d_out_kq, d_out_v = 3, 2, 1
sa = SelfAttention(d_in, d_out_kq, d_out_v)
print(sa(embedded_sentence))

## extended to 4 attention heads
torch.manual_seed(123)
block_size = embedded_sentence.shape[1]
mha = MultiHeadAttentionWrapper(
    d_in, d_out_kq, d_out_v, num_heads=4
)

context_vectors = mha(embedded_sentence)

print(context_vectors)
print("context_vectors.shape", context_vectors.shape)