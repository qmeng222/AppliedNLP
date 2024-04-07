#%% packages
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


import seaborn as sns
import matplotlib.pyplot as plt


#%% data import
twitter_file = '../data/Tweets.csv'
df = pd.read_csv(twitter_file).dropna()
df


# %% get class values based on categories
cat_id = {'neutral': 1,
         'negative': 0,
         'positive': 2}


df['class'] = df['sentiment'].map(cat_id)


#%% check the dataframe again after adding a new column
df


#%% Hyperparameters
BATCH_SIZE = 512
NUM_EPOCHS = 80


#%% separate independent and dependent features
X = df['text'].values # the input features or independent variables
y = df['class'].values # the target variable or dependent variables


#%% check X and y
print(f"X: {X.shape}, y: {y.shape}")
print(f"X: {X[:5]}, y: {y[:5]}")


# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=123) # ensure the data splitting is reproducible
print(f"X train: {X_train.shape}, y train: {y_train.shape}\nX test: {X_test.shape}, y test: {y_test.shape}")
print(X_test.shape, X_test[:5])


#%%
one_hot = CountVectorizer() # create an instance of the CountVectorizer class with default settings, using it to convert a collection of text documents into a matrix where each row corresponds to a document and each column corresponds to a unique word in the corpus


# learning the vocabulary (fitting) and transforming the documents into numerical representations (transforming):
X_train_onehot = one_hot.fit_transform(X_train)
# apply the transformation to the test data without re-learning the vocabulary:
X_test_onehot = one_hot.transform(X_test)


#%% check the one-hot encoded matrix:
print(f"X train one-hot: {X_train_onehot.shape}, X test one-hot: {X_test_onehot.shape}")


#%% Dataset Class
class SentimentData(Dataset):
   def __init__(self, X, y):
       super().__init__()
       self.X = torch.Tensor(X.toarray()) # sparse matrix format to dense tensor
       self.y = torch.Tensor(y).type(torch.LongTensor)
       self.len = len(self.X)


   # this method returns the length of the dataset:
   def __len__(self):
       return self.len


   # use this method to get a sample from the dataset at a specific index:
   def __getitem__(self, index):
       return self.X[index], self.y[index]


train_ds = SentimentData(X= X_train_onehot, y = y_train)
test_ds = SentimentData(X_test_onehot, y_test)


# %% Dataloader
train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=15000)


# %% Model
class SentimentModel(nn.Module):
   def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN = 10):
       super().__init__()
       self.linear = nn.Linear(NUM_FEATURES, HIDDEN)
       self.linear2 = nn.Linear(HIDDEN, NUM_CLASSES)
       self.relu = nn.ReLU()
       self.log_softmax = nn.LogSoftmax(dim=1)


   def forward(self, x):
       x = self.linear(x)
       x = self.relu(x)
       x = self.linear2(x)
       x = self.log_softmax(x)
       return x


#%% Model, Loss and Optimizer
model = SentimentModel(NUM_FEATURES = X_train_onehot.shape[1], NUM_CLASSES = 3)


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
sns.lineplot(x=list(range(len(train_losses))), y= train_losses)
plt.show()


# %% Prediction:
# temporarily disable gradient computation, which can significantly reduce memory usage and speed up computations during inference (when the model is used for prediction rather than training)：
with torch.no_grad():
   # a loop that iterates over batches of test data (X_batch contains input features, and y_batch contains corresponding labels) obtained from the test_loader：
   for X_batch, y_batch in test_loader:


       # pass the input features batch to the model to get the predicted log probabilities:
       y_test_pred_log = model(X_batch)


       # compute the predicted class labels by taking the index (argmax) of the maximum value along the specified dimension (dim=1, which corresponds to the class dimension). This converts the logits into class predictions.
       y_test_pred = torch.argmax(y_test_pred_log, dim = 1)


# %%
# remove any singleton dimensions using squeeze()
# then move the tensor to the CPU (if it's on a GPU)
# and finally convert the predicted class labels `y_test_pred` from PyTorch tensor to NumPy array using numpy()
y_test_pred_np = y_test_pred.squeeze().cpu().numpy()


# %%
# calculate the accuracy of the model's predictions using the accuracy_score function from scikit-learn:
acc = accuracy_score(y_pred=y_test_pred_np, y_true = y_test)
f"The accuracy of the model is {np.round(acc, 3)*100}%."


# %%
Counter(y_test).most_common() # [(1, 5635), (2, 4270), (0, 3835)]


#%%
# use `Counter` from the `collections` module to calculate the count of the most common class label in the ground truth labels `y_test`:
# `.most_common()` returns all elements ordered by their counts from highest to lowest
# [0] accesses the first tuple of the list returned by .most_common()
# [1] accesses the count of the most common element
most_common_cnt = Counter(y_test).most_common()[0][1]
# calculate the percentage of the most common class label in the test set (most_common_cnt / len(y_test)), rounds it to one decimal place, and formats it into a string:
print(f"Naive Classifier: {np.round(most_common_cnt / len(y_test) * 100, 1)} %")


# %% generate a heatmap of the confusion matrix using seaborn:
# visualize it as a heatmap with annotations (annot=True) and formatting (fmt=".0f") to display the counts without decimal places (use fmt="d" instead to format the numbers as integers):
sns.heatmap(confusion_matrix(y_test_pred_np, y_test), annot=True, fmt=".0f")
