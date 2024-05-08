#%% packages
from transformers import pipeline

#%% fill-mask
pipe = pipeline("fill-mask")
pipe("The capital of France is <mask>.")

# [{'score': 0.2703728973865509,
#   'token': 2201,
#   'token_str': ' Paris',
#   'sequence': 'The capital of France is Paris.'},
#  {'score': 0.05588363856077194,
#   'token': 12790,
#   'token_str': ' Lyon',
#   'sequence': 'The capital of France is Lyon.'},
#  {'score': 0.02989806793630123,
#   'token': 4612,
#   'token_str': ' Barcelona',
#   'sequence': 'The capital of France is Barcelona.'},
#  {'score': 0.02308163419365883,
#   'token': 12696,
#   'token_str': ' Monaco',
#   'sequence': 'The capital of France is Monaco.'},
#  {'score': 0.020979881286621094,
#   'token': 5459,
#   'token_str': ' Berlin',
#   'sequence': 'The capital of France is Berlin.'}]

# %%
