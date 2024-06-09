# %% packages
import nlpaug.augmenter.word as naw

# %%
text = "The quick brown fox jumps over the lazy dog."

# naw library > SynonymAug class > initialize an instance for specified augmentation action (remove words from the text):
aug = naw.RandomWordAug(action="crop")

# randomly removing words from the text:
augmented_text = aug.augment(text)
print("Original:")
print(text, "\n")  # The quick brown fox jumps over the lazy dog.
print("Augmented Text:")
print(augmented_text)  # ['The quick brown fox jumps dog.']

# %%
