# Attention

## Resources
1. https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention
2. https://theaisummer.com/attention/

## Overview

Starting with self attention we learn how an input is transformed to query, key and value components using the respective weight matrices. The follwing procedure of multiply queries x keys to obtain attention scores, which are then normalized with the softmax function to get attention weights. These weights are multiplied against the values matrix computed from the input and the result is a context vector for the input. The computation of a single context vector is done with a single head and this can be readily multiple heads. The process of multi head attention is simply computing context vectors for the input independently for each head. The result is then concatenated and returned. Finally, to ensure that the context vector does not contain predictions for future words within the input (ie. words in the input only depend on preceding words) then we can apply a mask. This is equivalent zeroing out the upper right corner of the context vector with normalization occuring before or after to enforce that each row in the context vector sums to 1. 