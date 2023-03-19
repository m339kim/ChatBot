import json
import torch
import random # to return random choice from possible pool of responses

from utils import bag_of_words, tokenize
from model import NeuralNet

# check if gpu device is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load and open files
with open('input.json', 'r') as f:
    intents = json.load(f)
FILE = "data.pth"
data = torch.load(FILE)

# load data from dictionary
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]

# load ffnn model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state) # knows learned params
model.eval()


# Chatbot UI
bot_name = "Darwin"
print("Welcome to Chatbot. 'exit' to exit")
while True:
    sentence = input("You: ")
    if sentence == "exit":
        break

    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words) # gets processed sentence
    x = x.reshape(1, x.shape[0]) # reshape to 1d data structure
    x = torch.from_numpy(x).to(device) # bow() returns numpy array

    output = model(x) # gives prediction
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()] # class label

    # check if answer probability is high enough
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # give response
    if prob.item() > 0.75: # check if prob is above lowerbound
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}\n")
    else:
        print(f"{bot_name}: I do not understand...\n") # probability too low