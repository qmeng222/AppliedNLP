#%% packages
from datasets import load_dataset
import pandas as pd # import the pandas library for data analysis
import numpy as np
import seaborn as sns # import the seaborn library for data visualization
import torch
from transformers import AutoModel, DistilBertTokenizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
# %% YELP Dataset
# source: https://huggingface.co/datasets/yelp_review_full
# train: 650k rows
# test: 50k rows
# features: ["label", "text"]
yelp = load_dataset('yelp_review_full')
yelp

#%% create dataset
train_ds = yelp['train'].select(range(1000)) # first 1000 rows

#%% Model and Tokenizer
model_name = 'distilbert-base-uncased'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model = AutoModel.from_pretrained(model_name).to(device)    # load the pre-trained weights for the specified model
tokenizer = DistilBertTokenizer.from_pretrained(model_name) # load the pre-trained tokenizer for the specified model

# %% Tokenizer
text = 'Hello, this is a sample sentence!'

encoded_text = tokenizer(text, return_tensors='pt')
encoded_text
# { "input_ids": [ [101, 7592, ..., 102], [101, 6813, ..., 102], ... ],
#   "attention_mask": [ [1, 1, ..., 0], [1, 1, ..., 0], ... ] }

# %% Tokens
# [101, 7592, ...] -> ["[CLS]", "hello", ",", "this", ..., "[SEP]"]
tokens = tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0])

# %%
# [ "[CLS]", "hello", ",", "this", ..., "[SEP]" ] -> "[CLS] hello, this is ... ! [SEP]"
tokenizer.convert_tokens_to_string(tokens)

# %% how large is the vocabulary?
tokenizer.vocab_size # 30522

# %% Max context length
max_context_length = tokenizer.model_max_length
max_context_length # 512

# %% Function for tokenization
def tokenize_text(batch):
    # padding: texts are filled with zeros based to longest example
    # truncation: texts are cut off after max_context_length
    return tokenizer(batch['text'], return_tensors='pt', padding='max_length', truncation=True)

# %%
# apply the function `tokenize_text` to each example in the dataset:
yelp_encodings = train_ds.map(tokenize_text, batched=True, batch_size=128)

# %%
yelp_encodings.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])  # include these cols in the PyTorch tensor

def get_last_hidden_state(batch):
    inputs = {k: v for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
        # [:, 0] refers to CLS token for complete sentence representation
    return {'hidden_state': last_hidden_state[:, 0]}

# %%
yelp_hidden_states = yelp_encodings.map(get_last_hidden_state, batched=True, batch_size=128)  # will have additional column 'hidden_state'
yelp_hidden_states

#%%
import joblib
joblib.dump(yelp_hidden_states, 'model/yelp_hidden_states.joblib')

#%%
# first 800 for training, rest for testing:
cutoff = 800
X_train = np.array(yelp_hidden_states['hidden_state'][:cutoff])
y_train = np.array(yelp_hidden_states['label'][:cutoff])
X_test = np.array(yelp_hidden_states['hidden_state'][cutoff: ])
y_test = np.array(yelp_hidden_states['label'][cutoff: ])
print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}") # (800, 768)
print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}") # (200, 768)

#%% Dummy model
# create an instance of the DummyClassifier class from scikit-learn & use the most frequent class label to make predictions:
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(X_train, y_train) # fit the dummy_model to the training data
dummy_model.score(X_test, y_test) # return the mean accuracy score of the classifier on the test data

# %% SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)
svm_model.score(X_test, y_test)

# %% Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_model.score(X_test, y_test)

# %%
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_model.score(X_test, y_test)
# %%
