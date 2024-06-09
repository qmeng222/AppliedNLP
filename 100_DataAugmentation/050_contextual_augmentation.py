# %% packages
import nlpaug.augmenter.word as naw

# %%
# specify the pre-trained BERT model we want to use for contextual word embeddings:
model_name = "bert-base-uncased"

# naw library >  ContextualWordEmbsAug ( replace words in the text with similar words based on their context) class > init an instance:
augmentation = naw.ContextualWordEmbsAug(model_path=model_name, action="substitute")

text_original = "The pen is mightier than the sword."
augmentation.augment(text_original)
# ['his pen is bigger than its sword.']
# ['that pen is mightier against our sword.']
