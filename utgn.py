"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import math
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import openprotein
from util import initial_pos_from_aa_string
from util import structures_to_backbone_atoms_padded
from util import get_backbone_positions_from_angular_prediction
from util import calculate_dihedral_angles_over_minibatch
from util import pass_messages
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from tape import ProteinBertModel, TAPETokenizer
from common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm, \
    _gen_bias_mask, _gen_timing_signal
import torch.nn.init as I

# seed random generator for reproducibility
torch.manual_seed(1)

NUM_DIHEDRALS = 3


def reduce_mean_angle(weights, angles):
    sins = torch.sin(angles)
    coss = torch.cos(angles)

    y_coords = torch.matmul(weights, sins)
    x_coords = torch.matmul(weights, coss)

    return torch.atan2(y_coords, x_coords)


class UTGN(openprotein.BaseModel):
    def __init__(self, dropout=0.5, alphabet_size=60, input_dim=42, num_vocab=256, n_hid=512, embedding_size=42,
                 n_head=8, n_layers=6,
                 use_gpu=False, batch_size=32):
        super().__init__(use_gpu, embedding_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_size
        self.use_gpu = use_gpu
        self.num_vocab = num_vocab
        self.batch_size = batch_size

        self.src_mask = None

        # self.emb = ProteinBertModel.from_pretrained('bert-base')

        self.W = nn.Linear(self.embedding_dim, self.num_vocab)

        self.pos_encoder = PositionalEncoding(num_vocab)

        encoder_layers = TransformerEncoderLayer(num_vocab, n_head, n_hid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        # initialize alphabet to random values between -pi and pi
        u = torch.distributions.Uniform(-3.14, 3.14)
        self.alphabet = nn.Parameter(u.rsample(torch.Size([alphabet_size, 3])))

        self._dehidrals = Dihedral(num_vocab, alphabet_size, self.batch_size)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _get_network_emissions(self, original_aa_string, pssm=-1):
        # set input
        packed_input_sequences = self.embed(original_aa_string, pssm)
        minibatch_size = int(packed_input_sequences[1][0])

        if self.src_mask is None or self.src_mask.size(0) != packed_input_sequences[1].size(0):
            mask = self._generate_square_subsequent_mask(packed_input_sequences[1].size(0)).to(self.device)
            self.src_mask = mask

        # transfomer
        state = F.relu(self.W(packed_input_sequences[0]))
        state, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(state, packed_input_sequences[1]))
        positional_encodings = self.pos_encoder(state)
        output = self.transformer_encoder(positional_encodings, self.src_mask)

        # convert internal representation to label
        output_angles = self._dehidrals(output, self.alphabet, )

        # coordinate
        backbone_atoms_padded, _ = get_backbone_positions_from_angular_prediction(output_angles, batch_sizes,
                                                                               self.use_gpu)

        return output_angles, backbone_atoms_padded, batch_sizes


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


class Dihedral(nn.Module):
    def __init__(self, num_vocab, alphabet_size, batch_size):
        super(Dihedral, self).__init__()
        self.num_vocab = num_vocab
        self.fully_connected_layer = nn.Linear(num_vocab, alphabet_size)
        self.prob = nn.Softmax(dim=1)
        self.batch_size = batch_size

    def forward(self, internal_representation, alphabet):
        linear = self.fully_connected_layer(internal_representation)
        flatten = torch.reshape(linear, [-1, 60])
        probs = self.prob(flatten / 1.0)
        dihedral_flatten = reduce_mean_angle(probs, alphabet)
        dihedral_flatten = dihedral_flatten.contiguous().view(internal_representation.size(0), self.batch_size,
                                                              NUM_DIHEDRALS)
        return dihedral_flatten
