# Attention

## Resources
1. https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention
2. https://theaisummer.com/attention/
3. https://arxiv.org/abs/1706.03762
4. https://arxiv.org/abs/1409.0473


## Overview

Starting with self attention we learn how an input is transformed to query, key and value components using the respective weight matrices. The follwing procedure of multiply queries x keys to obtain attention scores, which are then normalized with the softmax function to get attention weights. These weights are multiplied against the values matrix computed from the input and the result is a context vector for the input. The computation of a single context vector is done with a single head and this can be readily multiple heads. The process of multi head attention is simply computing context vectors for the input independently for each head. The result is then concatenated and returned. Finally, to ensure that the context vector does not contain predictions for future words within the input (ie. words in the input only depend on preceding words) then we can apply a mask. This is equivalent zeroing out the upper right corner of the context vector with normalization occuring before or after to enforce that each row in the context vector sums to 1. Masking the context vector is particularly useful when passing that context to a decoder to ensure that is does not leverage the future tokens when generating the output sequence.

## History

Sequence to Sequence learning was dominated by encoder-decoder models, primary among these were RNNs and LSTMs. These were vunerable to limitations when dealing with long sequences where previous information can be forgotten and vanishing gradients that failed to consider prior time steps. Attention was then born from the need to have a direct connection with the data at each timestamp.