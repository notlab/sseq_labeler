import torch
from torch import nn, autograd

import data

class LabellerNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, criterion=nn.NLLLoss(), lr=0.005):
        super(LabellerNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        self.criterion = criterion
        self.lr = lr

    def forward(self, ins, hidden):
        combined = torch.cat((ins, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return autograd.Variable(torch.zeros(1, self.hidden_size))
    
    def train(self, category_tensor, sample_tensor):
        hidden = self.init_hidden()

        self.zero_grad()

        for i in range(sample_tensor.size()[0]):
            output, hidden = self.__call__(sample_tensor[i], hidden)

        loss = self.criterion(output, category_tensor)
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in self.parameters():
            p.data.add_(-self.lr, p.grad.data)
                
        return output, loss.data[0]

class GeneratorNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return autograd.Variable(torch.zeros(1, self.hidden_size))

    def train(category_tensor, input_line_tensor, target_line_tensor):
        hidden = self.init_hidden()

        self.zero_grad()

        loss = 0

        for i in range(input_line_tensor.size()[0]):
            output, hidden = self.__call__(category_tensor, input_line_tensor[i], hidden)
            loss += criterion(output, target_line_tensor[i])

        loss.backward()

        for p in self.parameters():
            p.data.add_(-learning_rate, p.grad.data)

        return output, loss.data[0] / input_line_tensor.size()[0]

    
import time
import math

n_hidden = 128
n_iters = 100000
print_every = 500
plot_every = 100

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

categories, examples = data.inputs()
n_categories = len(categories)
rnn = LabellerNN(data.N_CHARS, n_hidden, n_categories, lr=0.01)

for i in range(1, n_iters + 1):
    category, sample, category_tensor, sample_tensor = data.random_sample(categories, examples)
    pred, loss = rnn.train(category_tensor, sample_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if i % print_every == 0:
        guess, guess_i = data.decode_prediction(categories, pred)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (i, i / n_iters * 100, time_since(start), loss, sample, guess, correct))

    # Add current loss avg to list of losses
    if i % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
