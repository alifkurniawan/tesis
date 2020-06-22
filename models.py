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
from tape import ProteinBertModel, TAPETokenizer
from common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm, \
    _gen_bias_mask, _gen_timing_signal
import torch.nn.init as I
# seed random generator for reproducibility
torch.manual_seed(1)


# sample model borrowed from
# https://github.com/lblaabjerg/Master/blob/master/Models%20and%20processed%20data/ProteinNet_LSTM_500.py
class ExampleModel(openprotein.BaseModel):
    def __init__(self, embedding_size, hidden_size, minibatch_size, use_gpu):
        super(ExampleModel, self).__init__(use_gpu, embedding_size)

        self.hidden_size = hidden_size
        self.num_lstm_layers = 2
        self.mixture_size = 500
        self.bi_lstm = nn.LSTM(self.get_embedding_size(), self.hidden_size,
                               num_layers=self.num_lstm_layers,
                               bidirectional=True, bias=True)
        self.hidden_to_labels = nn.Linear(self.hidden_size * 2,
                                          self.mixture_size, bias=True)  # * 2 for bidirectional
        self.init_hidden(minibatch_size)
        self.softmax_to_angle = SoftToAngle(self.mixture_size)
        self.soft = nn.LogSoftmax(2)
        self.batch_norm = nn.BatchNorm1d(self.mixture_size)

    def init_hidden(self, minibatch_size):
        # number of layers (* 2 since bidirectional), minibatch_size, hidden size
        initial_hidden_state = torch.zeros(self.num_lstm_layers * 2,
                                           minibatch_size, self.hidden_size)
        initial_cell_state = torch.zeros(self.num_lstm_layers * 2,
                                         minibatch_size, self.hidden_size)
        if self.use_gpu:
            initial_hidden_state = initial_hidden_state.cuda()
            initial_cell_state = initial_cell_state.cuda()
        self.hidden_layer = (autograd.Variable(initial_hidden_state),
                             autograd.Variable(initial_cell_state))

    def _get_network_emissions(self, original_aa_string):
        # set input
        packed_input_sequences = self.embed(original_aa_string)
        minibatch_size = int(packed_input_sequences[1][0])

        # set init
        self.init_hidden(minibatch_size)

        # bi-LSTM
        (data, bi_lstm_batches, _, _), self.hidden_layer = self.bi_lstm(packed_input_sequences, self.hidden_layer)

        # convert hidden to label
        emissions_padded, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(self.hidden_to_labels(data), bi_lstm_batches))
        emissions = emissions_padded.transpose(0, 1).transpose(1, 2)  # minibatch_size, self.mixture_size, -1
        emissions = self.batch_norm(emissions)
        emissions = emissions.transpose(1, 2)  # (minibatch_size, -1, self.mixture_size)
        probabilities = torch.exp(self.soft(emissions))
        output_angles = self.softmax_to_angle(probabilities).transpose(0, 1)  # max size, minibatch size, 3 (angles)

        # get backbond coordinates
        backbone_atoms_padded, _ = get_backbone_positions_from_angular_prediction(output_angles,
                                                                                  batch_sizes,
                                                                                  self.use_gpu)
        return output_angles, backbone_atoms_padded, batch_sizes


class SoftToAngle(nn.Module):
    def __init__(self, mixture_size):
        super(SoftToAngle, self).__init__()
        # Omega Initializer
        omega_components1 = np.random.uniform(0, 1, int(mixture_size * 0.1))  # set omega 90/10 pos/neg
        omega_components2 = np.random.uniform(2, math.pi, int(mixture_size * 0.9))
        omega_components = np.concatenate((omega_components1, omega_components2))
        np.random.shuffle(omega_components)

        phi_components = np.genfromtxt("data/mixture_model_pfam_"
                                       + str(mixture_size) + ".txt")[:, 1]
        psi_components = np.genfromtxt("data/mixture_model_pfam_"
                                       + str(mixture_size) + ".txt")[:, 2]

        self.phi_components = nn.Parameter(torch.from_numpy(phi_components)
                                           .contiguous().view(-1, 1).float())
        self.psi_components = nn.Parameter(torch.from_numpy(psi_components)
                                           .contiguous().view(-1, 1).float())
        self.omega_components = nn.Parameter(torch.from_numpy(omega_components)
                                             .view(-1, 1).float())

    def forward(self, x):
        phi_input_sin = torch.matmul(x, torch.sin(self.phi_components))
        phi_input_cos = torch.matmul(x, torch.cos(self.phi_components))
        psi_input_sin = torch.matmul(x, torch.sin(self.psi_components))
        psi_input_cos = torch.matmul(x, torch.cos(self.psi_components))
        omega_input_sin = torch.matmul(x, torch.sin(self.omega_components))
        omega_input_cos = torch.matmul(x, torch.cos(self.omega_components))

        eps = 10 ** (-4)
        phi = torch.atan2(phi_input_sin, phi_input_cos + eps)
        psi = torch.atan2(psi_input_sin, psi_input_cos + eps)
        omega = torch.atan2(omega_input_sin, omega_input_cos + eps)

        return torch.cat((phi, psi, omega), 2)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


class UTGN(openprotein.BaseModel):
    def __init__(self, embedding_size=10, num_vocab=20, transformer_dense_input=256, max_length=500, keep_prob=1.0,
                 n_layers=6, heads=8, ff_dims=512, embed_dims=256, use_gpu=False):
        self.embedding_size = embedding_size
        self.pretrained_model = ProteinBertModel.from_pretrained('bert-base')
        self.tokenizer = TAPETokenizer(vocab='iupac')
        self.transformer_encoder = Encoder(embedding_size=self.embedding_size, hidden_size=n_layers, num_heads=heads,
                                           total_key_depth=8, filter_size=25, max_length=max_length, act=True)

        self.W = nn.Linear(self.embedding_size, num_vocab)

        self.softmax = nn.Softmax(dim=1)

        ## POSITIONAL MASK
        self.mask = nn.Parameter(I.constant_(torch.empty(11, self.embedding_dim), 1))

    def _get_network_emissions(self, original_aa_string, pssms):

        return self._encoder_model(original_aa_string)


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, act=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        ## for t
        self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.num_layers = num_layers
        self.act = act
        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.proj_flag = False
        if (embedding_size == hidden_size):
            self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
            self.proj_flag = True

        self.enc = EncoderLayer(*params)

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        if (self.act):
            self.act_fn = ACT_basic(hidden_size)

    def forward(self, inputs):

        # Add input dropout
        x = self.input_dropout(inputs)

        if (self.proj_flag):
            # Project to hidden size
            x = self.embedding_proj(x)

        if (self.act):
            x, (remainders, n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal,
                                                     self.num_layers)
            return x, (remainders, n_updates)
        else:
            for l in range(self.num_layers):
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                x = self.enc(x)
            return x, None


# taken from https://github.com/andreamad8/Universal-Transformer-Pytorch
class ACT_basic(nn.Module):
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()
        step = 0
        # for l in range(self.num_layers):
        while (((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if (encoder_output):
                state, _ = fn((state, encoder_output))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = (
                    (state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop
            ## to save a line I assigned to previous_state so in the next
            ## iteration is correct. Notice that indeed we return previous_state
            step += 1
        return previous_state, (remainders, n_updates)
