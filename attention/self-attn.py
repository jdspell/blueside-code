import torch
import torch.nn as nn
import torch.nn.functional as F

sentence = 'Life is short, eat dessert first'

# create dictionary that maps each word to unique index
d = { s:i for i,s 
     in enumerate(sorted(sentence.replace(',', '').split()))}

print(d)

# encode the sentence from text to integers
# the results in a sentence that is made of the index of the word in the dictionary
sentence_int = torch.tensor(
    [d[s] for s in sentence.replace(',', '').split()]
)

print(sentence_int)


vocab_size = 50_000
torch.manual_seed(123)
# takes an argument for the size of the dictionary embedding (vocab size) and the size of each embedding vector
embed = torch.nn.Embedding(vocab_size, 3)
# the embedding functions as a lookup and for each word index from the sentence a unique vector will be returned
embedded_sentence = embed(sentence_int).detach()

print(embedded_sentence)
print(embedded_sentence.shape)


# self attention/scaled dot-product attention
## three weight matrices are used
## these break the inputs into three components (query, key and value)
## this is done by multiplying the inputs by each of the component weight matrices

## the input dimension is the size of each word vector
## note the number of elements in the value vector can be arbitrary

torch.manual_seed(123)
d = embedded_sentence.shape[1]
d_q, d_k, d_v = 2, 2, 4
## randomly initialize the weight matrices
W_query = torch.nn.Parameter(torch.rand(d, d_q))
W_key = torch.nn.Parameter(torch.rand(d, d_k))
W_value = torch.nn.Parameter(torch.rand(d, d_v))

## computing the attention vector for the second input element
x_2 = embedded_sentence[1]
## multiply the input by each of the attention component weight matrices
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2.shape)
print(key_2.shape)
print(value_2.shape)

## generalize to compute keys and values for all inputs
keys = embedded_sentence @ W_key
values = embedded_sentence @ W_value

print("keys.shape", keys.shape)
print("values.shape", values.shape)

## compute the unnormalized attention weights w (omega)
## w is the dot product between the query and key sequences (w = q * k)
omega_24 = query_2.dot(keys[4])
print(omega_24)

omega_2 = query_2 @ keys.T
print(omega_2)

## now we want to compute the normalized attention weights a (alpha)
## softmax will be used for normalization (a = w / sq(d_k))
attention_weights_2 = F.softmax(omega_2 / d_k**0.5, dim=0)
print(attention_weights_2)

## now we want to weight the original input vector with the new attention vector
## this is known as the context vector z (formula: z = a * values)
context_vector_2 = attention_weights_2 @ values
print(context_vector_2.shape)
print(context_vector_2)

## self attention class implementation
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
    
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T ## unnormalized attention weights
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1
        )

        context_vector = attn_weights @ values
        return context_vector


## the self attn class does the following:
## randomly initializes the parameters for the query, key, value weight matrices
## computes attn weights by multiplying the queries and keys then normalizing
## uses the attn weights from the value component of the input to get the context vector

torch.manual_seed(123)
#reduce d_out_v from 4 to 1, because we have 4 heads
d_in, d_out_kq, d_out_v = 3, 2, 4

sa = SelfAttention(d_in, d_out_kq, d_out_v)
# note the second row of the result matches context_vector_2 exactly
print(sa(embedded_sentence))

