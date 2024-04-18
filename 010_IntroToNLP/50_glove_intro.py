#%%
import torch
import torchtext # for preprocessing text data
import torchtext.vocab as vocab


# %%
# https://nlp.stanford.edu/projects/glove/
# load pre-trained word embeddings from the GloVe (Global Vectors for Word Representation) model:
glove = vocab.GloVe(name='6B', dim =100)


# %% number of words and embeddings
# retrieve the shape of the GloVe word vectors:
glove.vectors.shape

# glove.stoi["hope"] # 824
# glove.itos[824] # "hope"


#%% take a word as input, retrieve its embedding vector from the GloVe word vectors:
def get_embedding_vector(word):
   # `stoi`` stands for "string to index“:
   word_index = glove.stoi[word] # map each word to its corresponding index in the GloVe vocabulary
   emb = glove.vectors[word_index] # fetch the corresponding embedding vector
   return emb # return the embedding vector of the input word

print("👀", get_embedding_vector('chess'))
print(get_embedding_vector('chess').shape)


#%% take a word as input, calculate the distances between this word and all other words in the GloVe vocabulary based on their embedding vectors, and returns the max_n closest words along with their distances:
def get_closest_words_from_word(word, max_n=5):
   word_emb = get_embedding_vector(word) # retrieve the embedding vector for the input word


   # a list of tuples, where each tuple contains the word from the `glove.itos` list & its corresponding distance from the input word:
   distances = [(w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item()) for w in glove.itos] # calculate the Euclidean distance

   # sort the list of tuples (distances) based on the distances (second element of each tuple) in ascending order:
   dist_sort_filt = sorted(distances, key=lambda x: x[1])[:max_n]
   return dist_sort_filt

get_closest_words_from_word('chess')


#%% find closest words from embedding
def get_closest_words_from_embedding(word_emb, max_n=5):
   distances = [(w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item()) for w in glove.itos]
   dist_sort_filt = sorted(distances, key=lambda x: x[1])[:max_n]
   return dist_sort_filt


# %% find word analogies
# e.g. King is to Queen like Man is to Woman
def get_word_analogy(word1, word2, word3, max_n=5):
   # logic w1= king, ...
   # w1 - w2 + w3 --> w4
   word1_emb = get_embedding_vector(word1)
   word2_emb = get_embedding_vector(word2)
   word3_emb = get_embedding_vector(word3)
   word4_emb = word1_emb - word2_emb + word3_emb
   analogy = get_closest_words_from_embedding(word4_emb)
   return analogy

get_word_analogy(word1='sister', word2='brother', word3='nephew')
