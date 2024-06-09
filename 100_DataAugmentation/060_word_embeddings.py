# %%
import nlpaug.augmenter.word as naw

# !pip install gensim


# %%
# ref: https://github.com/stanfordnlp/GloVe
# initialize an instance of the WordEmbsAug class (for replacing words in the text with similar words based on their embeddings):
# NOTE: the `glove.6B.100d.txt` file (138MB) was downloaded from https://www.kaggle.com/datasets/sawarn69/glove6b100dtxt and saved under the same directory
augmentation = naw.WordEmbsAug(
    model_type="glove",  # GloVe (Global Vectors for Word Representation) embeddings model
    model_path="glove.6B.100d.txt",  # specify the txt file containing GloVe embeddings with 100 dimensions
    action="substitute",  # specift the augmentation action to be performed
)


# %%
text_original = "The pen is mightier than the sword."
augmentation.augment(text_original)  # 'The montlanc is blander than day sword.'
# %%
