import torch
import torch.nn as nn

# Feed Forward Neural Net with 2 hidden layers
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # 3 layers
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU() # activation function

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out) # relus in between
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no relus, no softmax here. will apply CrossEntropyLoss
        return out