import sys
sys.path.append('./src/')

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BASE=4


class View(nn.Module):

    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(self.target_shape)


# this layer drops out every channel at some positions, and keeps inputs at size 1
class SeqDropout(nn.Module):

    def __init__(self, input_dropout_freq):
        super().__init__()
        self.dropout = nn.Dropout2d(input_dropout_freq)

    def forward(self, x):
        y = self.dropout(x.permute(0, 2, 1)).permute(0, 2, 1)
        return y * (1 - self.dropout.p)


class GaussianNoise(nn.Module):

    def __init__(self, latent_noise_std):
        super().__init__()
        self.latent_noise_std = latent_noise_std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x)
            return x + noise * self.latent_noise_std
        return x


# this loss function combines binary cross entropy with neighbour distance
# neighbour distance is difference between latent variables at adjacent positions
# the output loss is weighted between BCE loss and this difference
class NeighbourDistanceLoss(nn.Module):

    def __init__(self, neighbour_loss_prop):
        super().__init__()
        self.neighbour_loss_prop = neighbour_loss_prop
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, x, z, y):
        if self.neighbour_loss_prop > 0.0:
            return self.bce_loss(z, x) * (1 - self.neighbour_loss_prop) + \
                self.mse_loss(y[:, :, :-1], y[:, :, 1:]) * self.neighbour_loss_prop
        return self.bce_loss(z, x)


class ReverseComplementConv1d(nn.Module):

    def __init__(self, in_channels, normal_out, complement_out, reverse_out, reverse_complement_out,
                kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        
        self.convs = nn.ModuleDict()
        if normal_out > 0:
            self.convs['normal'] = nn.Conv1d(in_channels, normal_out,
                                kernel_size, stride, padding, dilation, groups, bias)
        if complement_out > 0:
            self.convs['complement'] = nn.Conv1d(in_channels, complement_out,
                                kernel_size, stride, padding, dilation, groups, bias)
        if reverse_out > 0:
            self.convs['reverse'] = nn.Conv1d(in_channels, reverse_out,
                                kernel_size, stride, padding, dilation, groups, bias)
        if reverse_complement_out > 0:
            self.convs['reverse_complement'] = nn.Conv1d(in_channels, reverse_complement_out,
                                kernel_size, stride, padding, dilation, groups, bias)


    def forward(self, x):
        y_out = []
        if 'normal' in self.convs:
            y_out.append(self.apply_filter(x, self.convs['normal'], False, False))
        if 'complement' in self.convs:
            y_out.append(self.apply_filter(x, self.convs['complement'], True, False))
        if 'reverse' in self.convs:
            y_out.append(self.apply_filter(x, self.convs['reverse'], False, True))
        if 'reverse_complement' in self.convs:
            y_out.append(self.apply_filter(x, self.convs['reverse_complement'], True, True))
        return torch.cat(y_out, dim=1)  # concatenate along channel dimension


    def apply_filter(self, x, conv_filter, do_complement, do_reverse):
        y = []
        y.append(conv_filter(x))
        if do_complement:
            x_complement = torch.flip(x, [1])
            y.append(conv_filter(x_complement))
        if do_reverse:  # undo reverse along sequence so positions stay the same
            y.append(torch.flip(conv_filter(torch.flip(x, [2])), [2]))
        if do_complement and do_reverse:
            y.append(torch.flip(conv_filter(torch.flip(x_complement, [2])), [2]))
        y_out, index = torch.max(torch.stack(y, dim=0), dim=0)
        del index  # don't need this
        return y_out



class Autoencoder(nn.Module):


    def __init__(self, kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq, latent_noise_std, loss_fn):
        super().__init__()
        self.kernel_len = kernel_len
        self.latent_len = latent_len
        self.seq_len = seq_len
        self.seq_per_batch = seq_per_batch
        self.input_dropout_freq = input_dropout_freq
        self.latent_noise_std = latent_noise_std
        self.loss_fn = loss_fn

        # self.total_epochs = 0  # tracks number of epochs this model has been trained
        self.register_buffer('_total_batches', torch.tensor([[0]], dtype=torch.long))

        self.encode_layers = nn.ModuleDict()
        self.encode_layers['input_dropout'] = SeqDropout(input_dropout_freq)
        self.encode_layers['conv0'] = nn.Conv1d(N_BASE, latent_len, kernel_len)
        self.encode_layers['relu0'] = nn.ReLU()

        self.decode_layers = nn.ModuleDict()
        self.decode_layers['latent_noise'] = GaussianNoise(self.latent_noise_std)
        self.decode_layers['conv0'] = nn.ConvTranspose1d(latent_len, N_BASE, kernel_len)
        self.decode_layers['softmax'] = nn.Softmax(dim=1)


    # trims extra layers from the encoder portion, for transfer learning
    def decapitate(self, keep_n_layers=1, remove_n_layers=None):
        if remove_n_layers is None:
            remove_layers = list(self.encode_layers.keys())[keep_n_layers:]
        else:
            remove_layers = list(self.encode_layers.keys())[(-1 * remove_n_layers):]
        for key in remove_layers:
            _ = self.encode_layers.pop(key)
        self.decode_layers.clear()


    @property
    def total_batches(self):
        return self._total_batches.item()


    @total_batches.setter
    def total_batches(self, value):
        self._total_batches[0] = value


    def encode(self, x, override_convert_to_onehot=False):
        if not override_convert_to_onehot:
            x = F.one_hot(x, num_classes=N_BASE).permute(0, 2, 1).type(torch.float32)
        for layer in self.encode_layers.values():
            x = layer(x)
        return x


    def decode(self, y):
        for layer in self.decode_layers.values():
            y = layer(y)
        return y


    def forward(self, x, override_convert_to_onehot=False):
        latent = self.encode(x, override_convert_to_onehot)
        reconstructed = self.decode(latent)
        return reconstructed, latent


    def loss(self, x, y=None):
        x = F.one_hot(x, num_classes=N_BASE).permute(0, 2, 1).type(torch.float32)
        reconstructed, latent = self.forward(x, override_convert_to_onehot=True)
        if y is None:
            return self.loss_fn(x, reconstructed, latent)
        else:
            y = F.one_hot(y, num_classes=N_BASE).permute(0, 2, 1).type(torch.float32)
            return self.loss_fn(y, reconstructed, latent)


    def evaluate(self, x, true_x):
        x = F.one_hot(x, num_classes=N_BASE).permute(0, 2, 1).type(torch.float32)
        z, _ = self.forward(x, override_convert_to_onehot=True)
        predictions = torch.argmax(z, 1, keepdim=False)
        correct = (true_x == predictions)
        return {'correct': torch.sum(correct).item(), 'n_samples': correct.nelement()}


class MultilayerEncoder(Autoencoder):

    def __init__(self, kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq, latent_noise_std, loss_fn,
                hidden_len, pool_size, n_conv_and_pool, n_conv_before_pool, n_linear, hidden_dropout_freq):
        
        super().__init__(kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq,
                    latent_noise_std, loss_fn)

        pad = int(kernel_len / 2)
        sizes = [N_BASE] + [hidden_len * (i + 1) for i in range(n_conv_and_pool)]
        in_size = sizes[:-1]
        out_size = sizes[1:]

        encode_layers = nn.ModuleDict()
        encode_layers['input_dropout'] = SeqDropout(input_dropout_freq)
        for i, (n_in, n_out) in enumerate(zip(in_size, out_size)):
            encode_layers['conv{}0'.format(i)] = nn.Conv1d(
                    n_in, n_out, kernel_len, 1, pad)
            encode_layers['relu{}0'.format(i)] = nn.ReLU()
            for j in range(1, n_conv_before_pool):
                encode_layers['conv{}{}'.format(i, j)] = nn.Conv1d(
                        n_out, n_out, kernel_len, 1, pad)
                encode_layers['relu{}{}'.format(i, j)] = nn.ReLU()
            encode_layers['pool{}'.format(i)] = nn.MaxPool1d(pool_size)
            encode_layers['norm{}'.format(i)] = nn.BatchNorm1d(n_out)
            encode_layers['dropout{}'.format(i)] = nn.Dropout(hidden_dropout_freq)

        linear_size = int(seq_len / (pool_size ** n_conv_and_pool))
        encode_layers['view'] = View((-1, linear_size * out_size[-1]))
        encode_layers['linear0'] = nn.Linear(linear_size * out_size[-1], latent_len)
        for i in range(1, n_linear):
            encode_layers['reluL{}'.format(i)] = nn.ReLU()
            encode_layers['normL{}'.format(i)] = nn.BatchNorm1d(latent_len)
            encode_layers['linear{}'.format(i)] = nn.Linear(latent_len, latent_len)

        decode_layers = nn.ModuleDict()
        decode_layers['latent_noise'] = GaussianNoise(latent_noise_std)
        for i in range(1, n_linear):
            decode_layers['linear{}'.format(i)] = nn.Linear(latent_len, latent_len)
            decode_layers['reluL{}'.format(i)] = nn.ReLU()
            decode_layers['normL{}'.format(i)] = nn.BatchNorm1d(latent_len)
        decode_layers['linear0'] = nn.Linear(latent_len, linear_size * out_size[-1])
        decode_layers['view'] = View((-1, out_size[-1], linear_size))

        for i, (n_in, n_out) in enumerate(zip(reversed(in_size), reversed(out_size))):
            decode_layers['relu{}0'.format(i)] = nn.ReLU()
            decode_layers['norm{}'.format(i)] = nn.BatchNorm1d(n_out)
            decode_layers['pool{}'.format(i)] = nn.Upsample(scale_factor=pool_size,
                    mode='linear', align_corners=False)
            for j in reversed(range(1, n_conv_before_pool)):
                decode_layers['conv{}{}'.format(i, j)] = nn.ConvTranspose1d(
                        n_out, n_out, kernel_len, 1, pad)
                decode_layers['relu{}{}'.format(i, j)] = nn.ReLU()
            decode_layers['conv{}0'.format(i)] = nn.ConvTranspose1d(
                    n_out, n_in, kernel_len, 1, pad)
        decode_layers['softmax'] = nn.Softmax(dim=1)

        self.encode_layers = encode_layers
        self.decode_layers = decode_layers


class LatentLinearRegression(Autoencoder):

    def __init__(self, kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq, latent_noise_std, loss_fn,
                encoder, output_dim):
        super().__init__(kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq,
                    latent_noise_std, loss_fn)
        self.encoder = encoder
        self.linear = nn.Linear(latent_len, output_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()


    def forward(self, x):
        with torch.no_grad():
            latent = self.encoder.encode(x)
        y = self.linear(latent)
        return self.sigmoid(y)


    def loss(self, x, labels):
        y = self.forward(x)
        return self.loss_fn(y, labels.type(torch.float32))


    def evaluate(self, x, labels):
        y = self.forward(x)
        y_true = (y > 0.5)
        label_true = (labels > 0.5)
        y_false = torch.logical_not(y_true)
        label_false = torch.logical_not(label_true)
        true_pos = torch.sum(torch.logical_and(y_true, label_true)).item()
        true_neg = torch.sum(torch.logical_and(y_false, label_false)).item()
        false_pos = torch.sum(torch.logical_and(y_true, label_false)).item()
        false_neg = torch.sum(torch.logical_and(y_false, label_true)).item()
        return {'n_samples': y.nelement(),
                'true_pos': true_pos,
                'true_neg': true_neg,
                'false_pos': false_pos,
                'false_neg': false_neg}
