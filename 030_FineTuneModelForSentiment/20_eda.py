#%% packages
from huggingface_hub import list_datasets # for listing all the datasets scripts available on the Hugging Face Hub
from datasets import load_dataset # for loading files
import pandas as pd   # import the pandas library for data analysis
import seaborn as sns # import the seaborn library for data visualization

# %% list all the datasets scripts available on the Hugging Face Hub
datasets = list_datasets()
for dataset in datasets:
    print(dataset)

# %% YELP Dataset
# source: https://huggingface.co/datasets/yelp_review_full
# train: 650k rows
# test: 50k rows

yelp = load_dataset('yelp_review_full')
# %%
yelp

#%% create dataset
train_ds = yelp['train']
train_ds[0] # {"label": 4, "text": "dr. goldberg offers everything i look for ..."}

# %%
train_ds.features
# {
#     "label": ClassLabel(names=["1 star", "2 stars", ..., "5 stars"]),
#     "text": Value(dtype="string", id=None)
# }

# %% convert to dataframe (if necessary)
# train_ds.set_format('pandas')
# train_ds[:]

# %% imbalance of dataset
val_count = pd.DataFrame(train_ds['label']).value_counts() # calculate the count of each unique review value (1-5 stars) in the 'label' col

# create a count plot by taking the list of values from `val_count`:
sns.countplot(val_count.tolist()) # same count for each unique review value (balanced)

# %%
# count of words per review / per class

# create a DataFrame consists of two columns:
# each row of the DataFrame corresponds to a review text, with the 'review_length' indicating the number of words in the review and the 'label' indicating its associated label
df_review_len_label = pd.DataFrame({'review_length': [len(s.split()) for s in train_ds['text']], 'label': train_ds['label']})

# create a boxplot to visualize the distribution of review lengths ('review_length') for each label category ('label')
sns.boxplot(x='label', y='review_length', data=df_review_len_label)

# %%
