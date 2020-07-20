# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.2, kernel_size=1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # add option for convolution head
        self.kernel_size = kernel_size
        if self.kernel_size <= 1:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(ntoken, ninp, self.kernel_size),
            )
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        if self.kernel_size <= 1:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.kernel_size <= 1:
            src = self.encoder(src) * math.sqrt(self.ninp)
        else:
            # convolution needs (batch size, channels, sequence length)
            src = F.one_hot(src, num_classes=4).permute(1, 2, 0).float()
            # but transformer needs (sequence length, batch size, channels)
            src = self.encoder(src).permute(2, 0, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

import sys
sys.path.append('src/seq_util/')
from datasets import SequenceDataset, RandomRepeatSequence, print_target_vs_reconstruction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bptt = 200
batch_size = 10
eval_batch_size = 10
valid_split = 0.02
test_split = 0.01
dataset = SequenceDataset('data/ref_genome/test.fasta', seq_len=bptt, stride=bptt)
dataset = RandomRepeatSequence(bptt, 30000, 3, repeat_len=4)

valid_size = int(len(dataset) * valid_split)
test_size = int(len(dataset) * test_split)
train_data, valid_data = torch.utils.data.random_split(dataset, [len(dataset) - valid_size, valid_size])
valid_data, test_data = torch.utils.data.random_split(valid_data, [valid_size - test_size, test_size])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=eval_batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=2)


def get_batch(sequence, device, kernel_size=1):
    sequence = sequence.permute(1, 0)  # reorder to (sequence, batch) dimensions
    size = sequence.shape[1]  # batches
    data = sequence[0:-1]  # trim off the last element as target
    # first target is past the first conv kernel position
    target = sequence[kernel_size:].reshape((bptt - kernel_size) * size)  # have to reshape due to reordering
    return data.to(device), target.to(device)


def print_test_example(data, target, output, kernel_size=1):
    size = data.shape[1]  # batches
    print_target_vs_reconstruction(
        target.reshape((bptt - kernel_size, size))[::, 0].cpu(), F.softmax(output[::, 0], dim=1).cpu())


ntokens = 4 # the size of vocabulary
emsize = 50 # embedding dimension
nhid = 100 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
kernel_size = 1  # convolution layer as input to attention mechanism
epochs = 10 # The number of epochs
lr = 0.1  # learning rate
log_interval = 1000  # how often to log results

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout, kernel_size).to(device)

# weight for AGCT frequencies
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.35, 0.19, 0.18, 0.28]))
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.7)

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch, sequence in enumerate(train_loader):
        data, targets = get_batch(sequence, device, kernel_size=kernel_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            val_loss, n_correct, n_total = evaluate(model, valid_loader)
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | acc {:1.2f}'.format(
                    epoch, batch, len(train_loader), scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss), n_correct / n_total))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_loader):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    n_total, n_correct = 0, 0
    with torch.no_grad():
        do_print = True
        for sequence in data_loader:
            data, targets = get_batch(sequence, device, kernel_size=kernel_size)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += criterion(output_flat, targets).item()
            predictions = torch.argmax(output_flat, 1)
            incorrect = torch.nonzero(targets - predictions, as_tuple=False)
            n_correct += len(output_flat) - len(incorrect)
            n_total += len(output_flat)
            if do_print:
                do_print = False
                print_test_example(data, targets, output, kernel_size=kernel_size)
    return total_loss / len(data_loader), n_correct, n_total

best_val_loss = float("inf")
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss, n_correct, n_total = evaluate(model, valid_loader)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f} | acc {:1.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss), n_correct / n_total))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

test_loss, n_correct, n_total = evaluate(model, valid_loader)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | acc {:1.2f}'.format(
    test_loss, math.exp(test_loss), n_correct / n_total))
print('=' * 89)
