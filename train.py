import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import tokenize, stem, bag_of_words
from model import NeuralNet

# load json file
with open('input.json', 'r') as f:
    inputs = json.load(f) # now, 'inputs' is a dictionary

# init empty lists
all_words = [] 
tags = []
xy = [] # holds both pattern and tags as a tuple

# 1. Tokenize.
#   - Collect all words, patterns, tags
for input in inputs['intents']:
    tag = input['tag']
    tags.append(tag) # append tag to 'tags' array
    for pattern in input['patterns']: # iterate through patterns array
        w = tokenize(pattern) # tokenize pattern
        all_words.extend(w) # extend to 'all_words' (use 'extend' to avoid 2d array situation)
        xy.append((w, tag)) # sentence and the corresponding tag


ignore_punc = ['?', '!', '.', ','] # we wish to ignore these

# 2. Lowercase, stem. && 3. Elim punctuation chars
all_words = [stem(w) for w in all_words if w not in ignore_punc]
## sort, remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# 4. Convert to Bag of words
## training data
x_train = []
y_train = []

## Load processed data to x_train, y_train
for (pattern_sentence, tag) in xy: # unpack tuple
    # x: bag of words
    bag = bag_of_words(pattern_sentence, all_words) # pattern_sentence is tokenized
    x_train.append(bag)
    # y: labels
    label = tags.index(tag)
    y_train.append(label) # CrossEntropyLoss

## convert to numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)


# Create PyTorch dataset using x_train, y_train
# Create new dataset:
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    def __getitem__(self, index): # allows us to access dataset by index
        return self.x_data[index], self.y_data[index] # returns a tuple
    def __len__(self):
        return self.n_samples
    

dataset = ChatDataset()

# Hyperparams
batch_size = 8 
hidden_size = 8
output_size = len(tags) # num classes we have
input_size = len(all_words) # length of each bag of words
learning_rate = 0.001
num_epochs = 1000

train_loader = DataLoader(dataset=dataset, 
                          batch_size=batch_size, 
                          shuffle=True) # multi-processing for faster loads
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # check if gpu is available
model = NeuralNet(input_size, hidden_size, output_size).to(device) # push to device if available

# Loss and optimizer (to optimize learning rate)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        # push to device
        words = words.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad() # empty gradients
        loss.backward() # calculate back propagation
        optimizer.step()

    if (epoch + 1) % 100 == 0: # print every 100-th epoch step
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.5f}')

print(f'final loss={loss.item():.5f}')


# Save the data to pytorch file as a dict to `data.pth`
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}
FILE = "data.pth"
torch.save(data, FILE)
print(f'Training complete. Training file saved to {FILE}\n\n')