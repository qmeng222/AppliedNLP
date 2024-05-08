#%% packages
from transformers import pipeline

#%% translation
pipe = pipeline("translation_en_to_de")
pipe("The capital of France is Paris.")

# [{'translation_text': 'Die Hauptstadt Frankreichs ist Paris.'}]

# Verified with Google Translate:
# https://translate.google.com/?sl=de&tl=en&text=Die%20Hauptstadt%20Frankreichs%20ist%20Paris.&op=translate

# %%
