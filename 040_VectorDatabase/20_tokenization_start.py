#%% packages
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb # vector db
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from uuid import uuid4 # import the uuid4 function from the uuid (Universally Unique Identifier) module

# %% max_length
def max_token_length(txt_list:list):
    max_length = 0
    for txt in txt_list: # loop over each text string
        token_count = len(re.findall(r'\w+', txt)) # re.findall(pattern, string)
        """
        txt = "Hello, world! 123"
        re.findall(r'\w+', txt) would return ['Hello', 'world', '123']
        """
        if token_count > max_length:
            max_length = token_count
    return f"Max Token Length: {max_length} tokens"

# %% dummy text to play with:
lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Augue mauris augue neque gravida in fermentum et. Felis bibendum ut tristique et egestas quis ipsum suspendisse ultrices. Duis tristique sollicitudin nibh sit amet commodo nulla facilisi nullam. Pretium aenean pharetra magna ac placerat. Quis risus sed vulputate odio ut. Consectetur adipiscing elit duis tristique sollicitudin nibh. Nec nam aliquam sem et. Sed blandit libero volutpat sed cras. Faucibus pulvinar elementum integer enim neque volutpat ac. Mi in nulla posuere sollicitudin aliquam ultrices sagittis. Eget egestas purus viverra accumsan. Diam vel quam elementum pulvinar etiam non quam. Arcu cursus euismod quis viverra nibh cras. A scelerisque purus semper eget duis at. Lectus vestibulum mattis ullamcorper velit sed ullamcorper. Eget felis eget nunc lobortis mattis aliquam faucibus purus in. Elit scelerisque mauris pellentesque pulvinar pellentesque habitant. Ornare suspendisse sed nisi lacus sed. Interdum velit laoreet id donec ultrices. Ipsum a arcu cursus vitae congue mauris rhoncus aenean vel. Faucibus nisl tincidunt eget nullam non nisi. Urna condimentum mattis pellentesque id nibh. Tellus in hac habitasse platea dictumst vestibulum. Eget est lorem ipsum dolor. Enim eu turpis egestas pretium aenean pharetra magna ac placerat. Ac turpis egestas integer eget aliquet nibh. Vivamus arcu felis bibendum ut tristique et egestas. Nisi lacus sed viverra tellus in hac habitasse platea dictumst. Odio ut enim blandit volutpat maecenas volutpat. Turpis egestas sed tempus urna et pharetra pharetra massa. Dui nunc mattis enim ut tellus elementum sagittis vitae et. Nunc sed velit dignissim sodales ut eu. Aliquam ut porttitor leo a diam sollicitudin tempor id. At quis risus sed vulputate odio ut enim blandit volutpat. Gravida quis blandit turpis cursus in hac habitasse platea dictumst. Sit amet nulla facilisi morbi tempus iaculis urna. Diam maecenas sed enim ut sem viverra aliquet eget. Turpis egestas pretium aenean pharetra. At varius vel pharetra vel turpis nunc eget lorem. Integer quis auctor elit sed. Eget nunc lobortis mattis aliquam. Et magnis dis parturient montes nascetur ridiculus mus mauris vitae. Sollicitudin nibh sit amet commodo. Integer quis auctor elit sed vulputate mi sit amet mauris. Est placerat in egestas erat imperdiet. Ornare quam viverra orci sagittis eu volutpat odio facilisis mauris. Semper quis lectus nulla at volutpat diam. Amet volutpat consequat mauris nunc congue nisi. Ipsum nunc aliquet bibendum enim facilisis gravida neque convallis a. Et pharetra pharetra massa massa ultricies. Nunc eget lorem dolor sed viverra ipsum nunc aliquet bibendum."



# %% Sentence splitter
# chroma default sentence model "all-MiniLM-L6-v2"
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# For the all-MiniLM-L6-v2 (sentence-transformers) model, the input text longer than 256 word pieces is truncated by default:
model_max_chunk_length = 256

token_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=model_max_chunk_length, model_name="sentence-transformers/all-MiniLM-L6-v2", chunk_overlap=0)

# %% split the text
lorem_ipsum_token = token_splitter.split_text(lorem_ipsum)
print(lorem_ipsum_token)

# %% get max token length
max_token_length(lorem_ipsum_token) # less than 256

# %% Real Implemetation for large corpus (Bible)
# https://www.gutenberg.org/cache/epub/10/pg10.txt
# %% sneak peak of the text
text_path = "../data/bible.txt"
with open(text_path, "r", encoding="utf-8") as f:
    text_raw = f.read()

text_raw[:1000]

# %% Character splitter
character_splitter = RecursiveCharacterTextSplitter(
    separators=['\n    \n', '\n\n', '\n', '. '],
    chunk_size=1000,
    chunk_overlap=0
)

text_splitted = character_splitter.split_text(text_raw) # list of strings
print(f"Total # of splitted chunks: {len(text_splitted)}") # 4986

text_tokens = []
for text in text_splitted:
    text_tokens.extend(token_splitter.split_text(text))
print(f"Total # of tokens: {len(text_tokens)}") # 5363

# %% Check the token length
# reference: model card "By default, input text longer than 256 word pieces is truncated."
max_token_length(text_tokens)  # 216 tokens, didn't exceed the model limit

# %%
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)

# %% Size of embedding vector
len(embedding_fn(text_tokens[0])[0]) # 384

# %% initialize chromadb
chroma_db = chromadb.Client()
chroma_collection = chroma_db.create_collection("bible", embedding_function=embedding_fn)

# %% add all tokens to collection
ids = [str(uuid4()) for _ in range(len(text_tokens))]
ids[:5]

#%%
chroma_collection.add(documents=text_tokens, ids=ids)

# %% Run a query:
res = chroma_collection.query(query_texts=["What did Noah do?"], n_results=10)
res["documents"]

# %%
