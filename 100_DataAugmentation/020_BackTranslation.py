# %% import classes:
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


# %% Model and Tokenizer

# ref1: https://huggingface.co/facebook/m2m100_418M
# ref2: https://huggingface.co/docs/transformers/en/model_doc/m2m_100
# M2M100 is a multilingual encoder-decoder (seq-to-seq) model trained for Many-to-Many (M2M) multilingual translation.
# Load the model:
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
# Load the tokenizer:
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")


# %% 1. original language (`original_en`) -> target language (`target_de`):
original_en = "The pen is mightier than the sword."
original_encoded = tokenizer(original_en, return_tensors="pt")

generated_tokens = model.generate(
    **original_encoded,  # use `**`` operator to unpack the contents of `original_encoded`
    forced_bos_token_id=tokenizer.get_lang_id(
        "de"
    )  # specify the beginning of sequence (bos) should be forced to start with the token ID for German language
)

target_de = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
target_de  # ['Der Pen ist stärker als das Schwert.']

# %% 2. in reverse, target language (`target_de`) -> original language:
target_encoding = tokenizer(target_de, return_tensors="pt", padding=True)

generated_tokens = model.generate(
    **target_encoding, forced_bos_token_id=tokenizer.get_lang_id("en")
)

tokenizer.batch_decode(
    generated_tokens, skip_special_tokens=True
)  # ['The pen is stronger than the sword.']
