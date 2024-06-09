# %%
# pip install nlpaug
import nlpaug.augmenter.word as naw
import nltk

# %%
text = "The quick brown fox jumps over the lazy dog."

#  naw library > SynonymAug class > initialize an instance | performing synonym augmentation using synonyms from WordNet:
aug = naw.SynonymAug(aug_src="wordnet")

#  replace words in the text with their synonyms based on the augmentation strategy defined in the above SynonymAug instance:
augmented_text = aug.augment(text)
print("Original:")  # The quick brown fox jumps over the lazy dog.
print(text, "\n")
print("Augmented Text:")
print(augmented_text)  # ['The fast brownness fox jump out over the lazy dog.']

# %%
