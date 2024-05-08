#%% packages
from transformers import pipeline

# %% only provide task
# If no model is specified, the default model for the task + the default tokenizer of the model are selected:
pipe = pipeline(task="text-classification")

# %% run pipe
pipe("I like it very much.")
# [{'label': 'POSITIVE', 'score': 0.9998759031295776}]

# %% provide model (optional):
pipe = pipeline(task="text-classification",
                model="nlptown/bert-base-multilingual-uncased-sentiment")

# %%
# consume just a string:
pipe("I like it very much.")
# [{'label': '5 stars', 'score': 0.5000861883163452}]

# %% consume a list:
pipe(["I like it very much.",
      "I hate it."])
# [{'label': '5 stars', 'score': 0.5000861883163452},
#  {'label': '1 star', 'score': 0.7190805077552795}]

# %%
