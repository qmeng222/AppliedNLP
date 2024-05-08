#%% packages
from transformers import pipeline

# %% Question Answering
pipe = pipeline("question-answering")
pipe(context="The Big Apple is a nickname for New York City.", question="What is the Big Apple?")

# {'score': 0.6228488683700562,
#  'start': 17,
#  'end': 45,
#  'answer': 'a nickname for New York City'}

# %%
