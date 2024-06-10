# %%
from transformers import pipeline

# %%
# using the Transformers library to create a pipeline object for the "fill-mask" task:
# HF > Models > Natural Language Processing | Fill-Mask
# pipe = pipeline("fill-mask")
pipe = pipeline("fill-mask", model="bert-base-uncased")

pipe("Thank [MASK] for your [MASK]")

"""
[[{'score': 0.9909219145774841,
   'token': 2017,
   'token_str': 'you',
   'sequence': '[CLS] thank you for your [MASK] [SEP]'},
  {'score': 0.004243557807058096,
   'token': 2643,
   'token_str': 'god',
   'sequence': '[CLS] thank god for your [MASK] [SEP]'},
  {'score': 0.0009973165579140186,
   'token': 15003,
   'token_str': 'goodness',
   'sequence': '[CLS] thank goodness for your [MASK] [SEP]'},
  {'score': 0.0008035952923819423,
   'token': 2032,
   'token_str': 'him',
   'sequence': '[CLS] thank him for your [MASK] [SEP]'},
  {'score': 0.00031944605871103704,
   'token': 3071,
   'token_str': 'everyone',
   'sequence': '[CLS] thank everyone for your [MASK] [SEP]'}],

 [{'score': 0.26701804995536804,
   'token': 2393,
   'token_str': 'help',
   'sequence': '[CLS] thank [MASK] for your help [SEP]'},
  {'score': 0.25751832127571106,
   'token': 1012,
   'token_str': '.',
   'sequence': '[CLS] thank [MASK] for your. [SEP]'},
  {'score': 0.11682811379432678,
   'token': 2490,
   'token_str': 'support',
   'sequence': '[CLS] thank [MASK] for your support [SEP]'},
  {'score': 0.030073067173361778,
   'token': 6040,
   'token_str': 'advice',
   'sequence': '[CLS] thank [MASK] for your advice [SEP]'},
  {'score': 0.024611208587884903,
   'token': 2133,
   'token_str': '...',
   'sequence': '[CLS] thank [MASK] for your... [SEP]'}]]
"""
