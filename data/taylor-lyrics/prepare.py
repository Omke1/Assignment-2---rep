import os
import requests
import tiktoken
import numpy as np

# Define the locally stored dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input1.txt')
# or download the dataset from the web (but make sure it is in txt format)
if not os.path.exists(input_file_path):
    data_url = 'your-url-to-the-dataset'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

# Load the dataset
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# Create the bin folder "dataset" if it does not exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Split the data into training and validation sets
n = len(data) # number of characters in the dataset
train_data = data[:int(n*0.9)] # 90% of the data for training
val_data = data[int(n*0.9):] # 10% of the data for validation

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2") # use gpt2 bpe encoding
train_ids = enc.encode_ordinary(train_data) # encode the training data
val_ids = enc.encode_ordinary(val_data) # encode the validation data
print(f"The Train Dataset has {len(train_ids):,} tokens")
print(f"The Validation Dataset has {len(val_ids):,} tokens")

# Export the training and validation datasets in binary format for pytorch to use
train_ids = np.array(train_ids, dtype=np.uint16) # convert to numpy array
val_ids = np.array(val_ids, dtype=np.uint16) # convert to numpy array
train_ids.tofile('dataset/train.bin') # save the training data
val_ids.tofile('dataset/val.bin') # save the validation data
# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
