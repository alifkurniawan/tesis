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
    def __init__(self, dropout=0.5, alphabet_size=60, input_dim=20, num_vocab=256, n_hid=512, embedding_size=21,
                 n_head=8, n_layers=6,
                 use_gpu=False, batch_size=32, pretraining='bert-base', use_aa=True, use_pssm=True, use_token=False):
        super().__init__(use_gpu, embedding_size, pretraining)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_size
        self.use_gpu = use_gpu
        self.num_vocab = num_vocab
        self.batch_size = batch_size

        self.src_mask = None
        if pretraining is not -1:
            self.emb = ProteinBertModel.from_pretrained(pretraining)

        self.W = nn.Linear(self.embedding_dim, self.num_vocab)

        self.pos_encoder = PositionalEncoding(num_vocab)

        encoder_layers = TransformerEncoderLayer(num_vocab, n_head, n_hid, dropout)

        # self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        encoders = TransformerEncoder(encoder_layers, n_layers)
        self.transformer_encoder = UniversalTransformer(encoders, n_layers)

        # initialize alphabet to random values between -pi and pi
        u = torch.distributions.Uniform(-3.14, 3.14)
        self.alphabet = nn.Parameter(u.rsample(torch.Size([alphabet_size, 3])))

        self._dehidrals = Dihedral(num_vocab, alphabet_size, self.batch_size)
        self.use_aa = use_aa
        self.use_pssm = use_pssm
        self.use_token = use_token

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _get_network_emissions(self, original_aa_string, pssm, primary_token):

        # set input
        aa = original_aa_string if self.use_aa else -1
        evo = pssm if self.use_pssm else -1
        tok = primary_token if self.use_token else -1
        packed_input_sequences = self.embed(aa, evo, tok)
        minibatch_size = int(packed_input_sequences[1][0])

        if self.src_mask is None or self.src_mask.size(0) != packed_input_sequences[1].size(0):
            mask = self._generate_square_subsequent_mask(packed_input_sequences[1].size(0)).to(self.device)
            self.src_mask = mask

        # transfomer
        state = self.W(packed_input_sequences[0].to(self.device))
        state = F.relu(state)
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


class UniversalTransformer(nn.Module):
    def __init__(self, encoder_layers, n_layers, act_max_steps=10, keep_prob=1.0, act_threshold=0.5, num_vocab=256):
        super(UniversalTransformer, self).__init__()
        self.encoder_layers = encoder_layers
        self.n_layers = n_layers
        self.act_max_steps = act_max_steps
        self.keep_prob = keep_prob
        self.act_threshold = act_threshold
        self.num_vocab = num_vocab
        self.fc = nn.Linear(self.num_vocab, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.use_gpu = torch.cuda.is_available()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, state, input_mask):

        seq_length = state.size(0)
        batch_size = state.size(1)
        input_dim = state.size(2)
        step = 0
        halting_probability = torch.zeros(batch_size, seq_length, 1)
        remainders = torch.zeros(batch_size, seq_length, 1)
        n_updates = torch.zeros(batch_size, seq_length, 1)
        new_state = torch.zeros(batch_size, seq_length, 1)
        previous_state = torch.zeros(batch_size, seq_length, input_dim)

        if self.use_gpu:
            halting_probability = halting_probability.cuda()
            remainders = remainders.cuda()
            previous_state = previous_state.cuda()
            n_updates = n_updates.cuda()

        while self._should_continue(halting_probability, n_updates):
            (transformed_state, step, halting_probability, remainders,
             n_updates, new_state) = self._ut_function(state, step, halting_probability, remainders, n_updates,
                                                       previous_state,
                                                       self.encoder_layers, input_mask)

        return new_state

    def _should_continue(self, halting_probability, n_updates):
        return ((halting_probability < self.act_threshold) & (n_updates < self.act_max_steps)).byte().any()

    def _ut_function(self, state, step, halting_probability, remainders, n_updates, previous_state, encoder_layers,
                     input_mask):
        p = self.fc(state)
        p = self.sigmoid(p)
        p = p.transpose(0, 1)
        # Mask for inputs which have not halted yet
        still_running = (halting_probability < 1.0).float()

        # Mask of inputs which halted at this step
        new_halted = (halting_probability + p * still_running > self.act_threshold).float() * still_running

        # Mask of inputs which haven't halted, and didn't halt this step
        still_running = (halting_probability + p * still_running <= self.act_threshold).float() * still_running

        # Add the halting probability for this step to the halting
        # probabilities for those input which haven't halted yet
        halting_probability += p * still_running

        # Compute remainders for the inputs which halted at this step
        remainders += new_halted * (1 - halting_probability)

        # Add the remainders to those inputs which halted at this step
        halting_probability += new_halted * remainders

        # Increment n_updates for all inputs which are still running
        n_updates += still_running + new_halted

        # Compute the weight to be applied to the new state and output
        # 0 when the input has already halted
        # p when the input hasn't halted yet
        # the remainders when it halted this step
        update_weights = p * still_running + new_halted * remainders

        #transformed_state = state
        # for i in range(self.n_layers):
        transformed_state = self.encoder_layers(state, input_mask)

        transformed_state = transformed_state.transpose(0, 1)
        new_state = ((transformed_state * update_weights) + (previous_state *
                                                             (1 - update_weights)))

        new_state = new_state.transpose(0, 1)
        step += 1

        return (transformed_state, step, halting_probability, remainders,
                n_updates, new_state)


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
        print(linear.shape, dihedral_flatten.shape)
        dihedral_flatten = dihedral_flatten.contiguous().view(internal_representation.size(0), linear.size(1),
                                                              NUM_DIHEDRALS)
        return dihedral_flatten
