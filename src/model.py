import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch import nn, autograd, optim

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

class LabellerLSTM(nn.Module):

    def __init__(self, in_dim, lstm_dim, hid_dim, out_dim, criterion=nn.NLLLoss(), lr=0.005):
        super(LabellerLSTM, self).__init__()
        
        self.in_dim = in_dim
        self.lstm_dim = lstm_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        
        self.i2h_lstm = nn.LSTM(in_dim, lstm_dim)
        self.i2h = nn.Linear(in_dim, hid_dim)
        self.h2o = nn.Linear(lstm_dim + hid_dim, out_dim)
        self.softmax = nn.LogSoftmax()
        
        self.criterion = criterion
        self.lr = lr

    def forward(self, ins, hidden):
        lstm_out, hidden = self.i2h_lstm(ins, hidden)
        direct_out = self.i2h(ins)
        combined_out = torch.cat((lstm_out, direct_out), 2)
        output = self.h2o(combined_out).view(1, self.out_dim)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.lstm_dim)),
                autograd.Variable(torch.zeros(1, 1, self.lstm_dim)))
    
    def train(self, category_tensor, sample_tensor):
        self.zero_grad()
            
        hidden = self.init_hidden()
        optimizer = optim.SGD(self.parameters(), lr=self.lr)

        for i in range(sample_tensor.size()[0]):
            output, hidden = self.__call__(sample_tensor[i].view(1, 1, self.in_dim), hidden)

        output = output.view((1, self.out_dim))
        loss = self.criterion(output, category_tensor)
        loss.backward()
        optimizer.step()

        return output, loss.data[0]
            

class GeneratorNN(nn.Module):
    
    def __init__(self, n_categories, input_size, hidden_size, output_size, criterion=nn.NLLLoss(), lr=0.005):
        super(GeneratorNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()
        self.criterion = criterion
        self.lr = lr

    def forward(self, category, ins, hidden):
        input_combined = torch.cat((category, ins, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return autograd.Variable(torch.zeros(1, self.hidden_size))

    def train(self, category_tensor, input_line_tensor, target_line_tensor):
        hidden = self.init_hidden()

        self.zero_grad()

        loss = 0

        for i in range(input_line_tensor.size()[0]):
            output, hidden = self.__call__(category_tensor, input_line_tensor[i], hidden)
            loss += self.criterion(output, target_line_tensor[i])

        loss.backward()

        for p in self.parameters():
            p.data.add_(-self.lr, p.grad.data)

        return output, loss.data[0] / input_line_tensor.size()[0]


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def evaluate(model, sample_tensor, reshape_for_minibatch=False):
    hidden = model.init_hidden()
    
    for i in range(sample_tensor.size()[0]):
        if reshape_for_minibatch:
            pred, hidden = model(sample_tensor[i].view(1, 1, model.in_dim), hidden)
        else:
            pred, hidden = model(sample_tensor[i], hidden)

    return pred

def train_labeller():
    n_hidden = 128
    n_iters = 100000
    print_every = 500
    plot_every = 100

    current_loss = 0
    all_losses = []

    categories, examples = data.inputs()
    n_categories = len(categories)
    rnn = LabellerNN(data.N_CHARS, n_hidden, n_categories, lr=0.01)

    start = time.time()

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

    plot_losses(all_losses, save=True, fname='simple_rnn_losses.png')
    plot_confusion(rnn, categories, examples, fname='simple_rnn_confusion.png', reshape_for_minibatch=False)

def train_lstm():
    n_hidden = 128
    n_iters = 100000
    print_every = 500
    plot_every = 100

    current_loss = 0
    all_losses = []

    categories, examples = data.inputs()
    n_categories = len(categories)
    lstm = LabellerLSTM(data.N_CHARS, 64, 64, n_categories, lr=0.1)

    start = time.time()

    for i in range(1, n_iters + 1):
        category, sample, category_tensor, sample_tensor = data.random_sample(categories, examples)
        pred, loss = lstm.train(category_tensor, sample_tensor)
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

    plot_losses(all_losses, save=True, fname='lstm_losses.png')
    plot_confusion(lstm, categories, examples, fname='lstm_confusion.png', reshape_for_minibatch=True)
    

def plot_losses(all_losses, save=True, fname='losses'):
    plt.figure(figsize=(20, 20))

    if save:
        plt.plot(all_losses)
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.plot(all_losses)
        plt.show()

def plot_confusion(model, categories, examples, fname='confusion.png', reshape_for_minibatch=False):
    n_categories = len(categories)
    confusion = torch.zeros(n_categories, n_categories)
    
    for i in range(10000):
        category, sample, category_tensor, sample_tensor = data.random_sample(categories, examples)
        log_logits = evaluate(model, sample_tensor, reshape_for_minibatch=reshape_for_minibatch)
        pred, pred_i = data.decode_prediction(categories, log_logits)
        category_i = categories.index(category)
        confusion[category_i][pred_i] += 1

    # Normalize
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    ax.set_xticklabels([''] + categories, rotation=90)
    ax.set_yticklabels([''] + categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(fname, bbox_inches='tight')


def train_generator():
    n_iters = 100000
    print_every = 500
    plot_every = 500

    all_losses = []
    total_loss = 0

    categories, examples = data.inputs()
    n_categories = len(categories)
    rnn = GeneratorNN(n_categories, data.N_CHARS, 128, data.N_CHARS)

    start = time.time()

    for i in range(1, n_iters + 1):
        category, sample, category_tensor, sample_tensor = data.random_sample(categories, examples, one_hot_categories=True)
        target_tensor = autograd.Variable(data.target_encode(sample))
        output, loss = rnn.train(category_tensor, sample_tensor, target_tensor)
        total_loss += loss

        if i % print_every == 0:
            print('%s (%d %d%%) %.4f' % (time_since(start), i, i / n_iters * 100, loss))
            
        if i % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0
        
train_labeller()
#train_lstm()
#train_generator()
