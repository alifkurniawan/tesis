import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

torch.manual_seed(1)


class RgnModel(nn.Module):
    def __init__(self, embedding_size, minibatch_size, hidden_size=800, linear_units=20, use_gpu=False):
        super(RgnModel, self).__init__()
        self.num_lstm_layers = 2
        self.hidden_size = hidden_size
        self.bi_lstm = nn.LSTM(self.get_embedding_size(), hidden_size=self.hidden_size, num_layers=self.num_lstm_layers,
                               bidirectional=True)
        # initialize alphabet to random values between -pi and pi
        u = torch.distributions.Uniform(-3.14, 3.14)
        self.alphabet = nn.Parameter(u.rsample(torch.Size([linear_units, 3])))
        self.linear = nn.Linear(hidden_size * 2, linear_units)

        # set coordinates for first 3 atoms to identity matrix
        self.A = torch.tensor([0., 0., 1.])
        self.B = torch.tensor([0., 1., 0.])
        self.C = torch.tensor([1., 0., 0.])

        # bond length vectors C-N, N-CA, CA-C
        self.avg_bond_lens = torch.tensor([1.329, 1.459, 1.525])
        # bond angle vector, in radians, CA-C-N, C-N-CA, N-CA-C
        self.avg_bond_angles = torch.tensor([2.034, 2.119, 1.937])

    def forward(self, sequences, lengths):
        max_len = sequences.size(0)
        batch_sz = sequences.size(1)
        lengths = torch.tensor(lengths, dtype=torch.long, requires_grad=False)
        order = [x for x, y in sorted(enumerate(lengths), key=lambda x: x[1], reverse=True)]
        conv = zip(range(batch_sz), order)  # for unorder after LSTM

        # add absolute position of residue to the input vector
        abs_pos = torch.tensor(range(max_len), dtype=torch.float32).unsqueeze(1)
        abs_pos = (abs_pos * torch.ones((1, batch_sz))).unsqueeze(2)  # broadcasting

        h0 = Variable(torch.zeros((self.num_layers * 2, batch_sz, self.hidden_size)))
        c0 = Variable(torch.zeros((self.num_layers * 2, batch_sz, self.hidden_size)))

        # input needs to be float32 and require grad
        sequences = torch.tensor(sequences, dtype=torch.float32, requires_grad=True)
        pad_seq = torch.cat([sequences, abs_pos], 2)

        packed = pack_padded_sequence(pad_seq[:, order], lengths[order], batch_first=False)

        lstm_out, _ = self.lstm(packed, (h0, c0))
        unpacked, _ = pad_packed_sequence(lstm_out, batch_first=False, padding_value=0.0)

        # reorder back to original to match target
        reorder = [x for x, y in sorted(conv, key=lambda x: x[1], reverse=False)]
        unpacked = unpacked[:, reorder]

        # for example, see https://bit.ly/2lXJC4m
        softmax_out = F.softmax(self.linear(unpacked), dim=2)
        sine = torch.matmul(softmax_out, torch.sin(self.alphabet))
        cosine = torch.matmul(softmax_out, torch.cos(self.alphabet))
        out = torch.atan2(sine, cosine)

        # create as many copies of first 3 coords as there are samples in the batch
        broadcast = torch.ones((batch_sz, 3))
        pred_coords = torch.stack([self.A * broadcast, self.B * broadcast, self.C * broadcast])

        for ix, triplet in enumerate(out[1:]):
            pred_coords = geometric_unit(pred_coords, triplet,
                                         self.avg_bond_angles,
                                         self.avg_bond_lens)
        # pred_coords.register_hook(self.save_grad('pc'))

        # pdb.set_trace()
        return pred_coords

    def save_grad(self, name):
        def hook(grad): self.grads[name] = grad

        return hook
