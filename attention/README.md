# Attention

## Resources
1. https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention
2. https://theaisummer.com/attention/
3. https://arxiv.org/abs/1706.03762
4. https://arxiv.org/abs/1409.0473
5. https://lilianweng.github.io/posts/2018-06-24-attention/


## Overview

Starting with self attention we learn how an input is transformed to query, key and value components using the respective weight matrices. The follwing procedure of multiply queries x keys to obtain attention scores, which are then normalized with the softmax function to get attention weights. These weights are multiplied against the values matrix computed from the input and the result is a context vector for the input. The computation of a single context vector is done with a single head and this can be readily multiple heads. The process of multi head attention is simply computing context vectors for the input independently for each head. The result is then concatenated and returned. Finally, to ensure that the context vector does not contain predictions for future words within the input (ie. words in the input only depend on preceding words) then we can apply a mask. This is equivalent zeroing out the upper right corner of the context vector with normalization occuring before or after to enforce that each row in the context vector sums to 1. Masking the context vector is particularly useful to ensure that future tokens are not leveraged when generating the output sequence.

## History

Sequence to Sequence learning was dominated by encoder-decoder architectures linked by a single context vector. The encoder/decoder models were typically RNNs and LSTMs. The models were vunerable to limitations when dealing with long sequences where previous information can be forgotten and vanishing gradients that failed to consider prior time steps. Additionally, a vulnerability of the encoder-decoder architecture was that the context vector was fixed and portions of the input might not be attended to. Attention was then born from the need to have a direct connection between the context vector and the entire input, which allows for weighting for each output item.

## Intuition
- Attention is about finding alignment between pieces of an input. Computation of attention scores allows for seeing how 'aligned' two pieces of input are. Additionally, a portion of the input might align with one or many other portions of the input (the mapping can be one to many).
- Computational complexity as sequence length is scaled is quadratic.
- Connection between the context vector and the entire input is needed to prevent forgetting portions of the input sequence when generating outputs.