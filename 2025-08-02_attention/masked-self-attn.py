import torch
import torch.nn as nn
import torch.nn.functional as F

from self_attn import embedded_sentence

## masked self attention is as referred to as causal self attn
## it forces the prediction of the next word to only depend on the preceding words
## this is acheived by masking the attention weights to hide the future tokens


## self attention recap
torch.manual_seed(123)

d_in, d_out_kq, d_out_v = 3, 2, 4

W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
W_key = nn.Parameter(torch.rand(d_in, d_out_kq))
W_value = nn.Parameter(torch.rand(d_in, d_out_v))

x = embedded_sentence

keys =  x @ W_key
queries = x @ W_query
values = x @ W_value

## attn scores are the omegas
## the unnormalized attention weights
attn_scores = queries @ keys.T

print(attn_scores)
print(attn_scores.shape)

## compute the normalized weights
## note this results in each row summing to 1
attn_weights = torch.softmax(attn_scores / d_out_kq**0.5, dim=1)
print(attn_weights)

## we will create a matrix (mask) that has 1s on the diagonal and below the diagonal. The upper cells will contain 0s.
block_size = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(block_size, block_size))
print(mask_simple)

## then zero out the upper triangle of the attn weights
masked_simple = attn_weights*mask_simple
print(masked_simple)

## after performing the mask the rows no longer sum to 1 like they did post normalization
## we can rescale the rows to re-normalize so that they sum to 1
row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

## current masking flow (attention scores -- softmax --> attention weights -- diagonal mask --> masked attention scores -- normalize --> masked attention weights)
## an alternative masking procedure would be to mask prior to performing the softmax
## then can be done by using negative infinity above the diagonal instead of zero like with the previous procedure
mask = torch.triu(torch.ones(block_size, block_size), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

## then just apply softmax to get the normalized masked attention weights which treats the -inf values as 0
attn_weights = torch.softmax(masked / d_out_kq**0.5, dim=1)
print(attn_weights)