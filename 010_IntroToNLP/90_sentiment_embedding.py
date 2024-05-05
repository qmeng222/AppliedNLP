#%%
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer # for generating vector embeddings of sentences

#%% import data
twitter_file = '../data/Tweets.csv'
df = pd.read_csv(twitter_file).dropna()
# read the CSV file into a pandas DataFrame
# then drop any rows with missing values (NaNs) from the DataFrame
df

#%% Create Target Variable
cat_id = {'negative': 0,
          'neutral': 1,
          'positive': 2}

df['class'] = df['sentiment'].map(cat_id)
df

#%% Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 80
MAX_FEATURES = 10

#%%
# load a pre-trained sentence embedding model from the Hugging Face Transformers library
emb_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v1')

# this model maps sentences and paragraphs to a 768 dimensional dense vector space:
sentences = [ "Each sentence is converted"]
embeddings = emb_model.encode(sentences)
print(embeddings.shape) # (1, 768)
print(embeddings.squeeze().shape) # squeeze() function removes any singleton dimensions from the shape of the embeddings

#%% prepare X and y
# # X is decoded data:
# X = emb_model.encode(df['text'].values)

# # (with statement ensures that the file is properly closed after writing) open the file named "tweets_X.pkl" in binary write mode ("wb") for writing the encoded data:
# with open("../data/tweets_X.pkl", "wb") as output_file:
#     # use the pickle.dump() function to serialize the X object (which contains the encoded data) and write it to the opened file (output_file), this allows the data to be saved in a binary format that can be easily read back later:
#     pickle.dump(X, output_file) # serialize

# open the file "tweets_X.pkl" in binary read mode ("rb") for reading the encoded data:
with open("../data/tweets_X.pkl", "rb") as input_file:
    # deserialize (to reverse the process and reconstruct the original object):
    X = pickle.load(input_file)
X.shape # (27480, 768)

y = df['class'].values
y[:5] # [1, 0, 0, 0, 0]

#%% train test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123) # by setting a specific value, you ensure that the data split will be reproducible

# %%
# define a class that inherits from the Dataset class provided by PyTorch:
class SentimentData(Dataset):
    # init the dataset object with the input features X and labels y:
    def __init__(self, X, y):
        super().__init__()
        # convert the input features X into a PyTorch tensor and assigns it to the X attribute of the dataset object:
        self.X = torch.Tensor(X)
        # convert the labels y into a PyTorch tensor and ensures that it has the data type torch.LongTensor:
        self.y = torch.Tensor(y).type(torch.LongTensor)
        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# create an instance of the SentimentData class for the training data (X_train and y_train):
train_ds = SentimentData(X= X_train, y = y_train)
# create an instance of the SentimentData class for the test data (X_test and y_test):
test_ds = SentimentData(X_test, y_test)

# %% Dataloaders:
train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=15000)

# %%
# define a class that inherits from the nn.Module class provided by PyTorch:
class SentimentModel(nn.Module):
    # init the model object with the specified number of input features (NUM_FEATURES), output classes (NUM_CLASSES), and hidden units (HIDDEN, default value is 10)：
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN = 10):
        # call the constructor of the parent class (nn.Module) to initialize its attributes：
        super().__init__()

        # define a linear transformation layer (nn.Linear) that maps input features to a hidden layer：
        self.linear = nn.Linear(NUM_FEATURES, HIDDEN)

        # map the hidden layer to the output layer：
        self.linear2 = nn.Linear(HIDDEN, NUM_CLASSES)

        # Rectified Linear Unit (ReLU) AF to introduce non-linearity to the model:
        self.relu = nn.ReLU()

        # convert raw output scores into log probabilities along the second dimension (across the output classes):
        self.log_softmax = nn.LogSoftmax(dim=1)

    # forward pass (define how input data is processed through the model):
    def forward(self, x):
        x = self.linear(x) # apply the first linear transformation to the input data
        x = self.relu(x) # apply the ReLU AF to the output of the first linear transformation
        x = self.linear2(x) # apply the second linear transformation to the output of the ReLU activation
        x = self.log_softmax(x) # apply the Log Softmax activation function to the output of the second linear transformation
        return x # return the final output of the model

#%% Model, Loss and Optimizer
model = SentimentModel(NUM_FEATURES = X_train.shape[1], NUM_CLASSES = 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

# %% Model Training
train_losses = []
for e in range(NUM_EPOCHS):
    curr_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred_log = model(X_batch)
        loss = criterion(y_pred_log, y_batch.long())

        curr_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_losses.append(curr_loss)
    print(f"Epoch {e}, Loss: {curr_loss}")

# %%
# create a line plot using the lineplot function from the seaborn library (sns):
sns.lineplot(x=list(range(len(train_losses))), y= train_losses)

# %% Model Evaluation
# do not track the gradients during evaluation:
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_test_pred_log = model(X_batch)
        y_test_pred = torch.argmax(y_test_pred_log, dim = 1)

# %%
y_test_pred_np = y_test_pred.squeeze().cpu().numpy()

# %%
acc = accuracy_score(y_pred=y_test_pred_np, y_true = y_test)
f"The accuracy of the model is {np.round(acc, 3)*100}%."
# %%
# create a Counter object:
# most_common() is a method of the Counter object that returns a list of tuples, where each tuple contains an element and its count, sorted in descending order of counts
most_common_cnt = Counter(y_test).most_common()[0][1]
print(f"Naive Classifier: {np.round(most_common_cnt / len(y_test) * 100, 1)} %")
# %%