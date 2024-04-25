#%% packages
import pandas as pd
from plotnine import ggplot, aes, geom_text, labs
from sklearn.manifold import TSNE
import torchtext.vocab as vocab
import torch

#%% load pre-trained word embeddings from the GloVe model:
glove_dim = 100
glove = vocab.GloVe(name='6B', dim = glove_dim)

#%% input word (str) -> index in Glove vocabulary -> embedding vector of the input word:
def get_embedding_vector(word):
    word_index = glove.stoi[word] # string to index
    emb = glove.vectors[word_index] # retrieve the embedding vector
    return emb

def get_closest_words_from_word(word, max_n=5):
    word_emb = get_embedding_vector(word)
    # a list of tuples, where each tuple contains the word from the `glove.itos` list & its corresponding distance (Euclidean distance or cosine similarity) from the input word:
    distances = [(w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item()) for w in glove.itos] # idx to str
    # sort the list of tuples based on the second element of each tuple:
    dist_sort_filt = sorted(distances, key=lambda x: x[1])[:max_n]
    return [item[0] for item in dist_sort_filt] # return a list containing only the words

get_closest_words_from_word(word='chess', max_n=10)

#%% set up data structures for storing and organizing words and their associated categories:
words = []
categories = ["numbers", "algebra", "music", "science", "technology"]
df_word_cloud = pd.DataFrame({
    "category": [], # 1st col
    "word": [] # 2nd col
})

for category in categories:
    # print("👀", category) # numbers  algebra  music  science  technology
    # retrieve the 20 closest words associated with that category:
    closest_words = get_closest_words_from_word(word=category, max_n=20)

    # create a temporary DataFrame:
    temp = pd.DataFrame({
        "category": [category] * len(closest_words),
        "word": closest_words
    })

    # vertically concatenate `temp` DataFrame to `df_word_cloud`:
    df_word_cloud = pd.concat([df_word_cloud, temp], ignore_index=True) # reset the idx of the resulting DataFrame to start from 0
    # print(df_word_cloud)

# %%
n_rows = df_word_cloud.shape[0] # number of rows in the `df_word_cloud` DataFrame
n_cols = glove_dim # the dimensionality of the GloVe embeddings
X = torch.empty((n_rows, n_cols)) # initialize an empty pytorch tensor
for i in range(n_rows): # iterate over each row (word) in the `df_word_cloud` DataFrame
    current_word = df_word_cloud.loc[i, "word"] # retrieve the word (from the "word" column) at the i-th row
    X[i, :] = get_embedding_vector(current_word) # assign the embedding vector of `current_word` to the i-th row of tensor `X`
    print(f"👀{i}: {current_word}") # print idx i and the current word being processed

# %% visualize the GloVe word embeddings in a 2D space:

tsne = TSNE(n_components=2) # initialize a t-SNE object with the desired number of components (2, for 2D visualization)
X_tsne = tsne.fit_transform(X.cpu().numpy())
# tensor -> array, t-SNE only operates on NumPy arrays
# fit the t-SNE model to the data (X) and transforms it to the reduced 2D space

# add new columns, "x" and "y", to the `df_word_cloud` DataFrame:
df_word_cloud["x"] = X_tsne[:, 0]
df_word_cloud["y"] = X_tsne[:, 1]

ggplot(data=df_word_cloud.sample(25)) + aes(x = "x", y = "y", label = "word", color = "category") + geom_text() + labs(title="GloVe Word Embeddings and Categories")
# `ggplot`: initialize a ggplot object
# `aes(x = "x", y = "y", label = "word", color = "category")`: define aesthetic mappings, where "x" and "y" are the coordinates, "label" is the word label, and "color" represents the category
# `geom_text()`: add text to the plot using the coordinates specified

# %%
