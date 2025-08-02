import torch 

sentence = 'here is my fake sentence'

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
embed = torch.nn.Embedding(vocab_size, 3)
embedded_sentence = embed(sentence_int).detach()

print(embedded_sentence)
print(embedded_sentence.shape)

