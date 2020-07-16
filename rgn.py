"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
import openprotein
from utgn import Dihedral
from util import initial_pos_from_aa_string
from util import structures_to_backbone_atoms_padded
from util import get_backbone_positions_from_angular_prediction
from util import calculate_dihedral_angles_over_minibatch
from util import pass_messages

# seed random generator for reproducibility
torch.manual_seed(1)


# sample model borrowed from
# https://github.com/lblaabjerg/Master/blob/master/Models%20and%20processed%20data/ProteinNet_LSTM_500.py
class RGN(openprotein.BaseModel):
    def __init__(self, embedding_size, minibatch_size, use_gpu, num_vocab=256, alphabet_size=60, pretraining='bert-base'):
        super(RGN, self).__init__(use_gpu, embedding_size, pretraining)

        self.hidden_size = 800
        self.num_lstm_layers = 2
        self.mixture_size = 256
        u = torch.distributions.Uniform(-3.14, 3.14)
        self.alphabet = nn.Parameter(u.rsample(torch.Size([alphabet_size, 3])))
        self.embedding_size = embedding_size
        self.bi_lstm = nn.LSTM(self.get_embedding_size(), self.hidden_size,
                               num_layers=self.num_lstm_layers,
                               bidirectional=True, bias=True)
        self.hidden_to_labels = nn.Linear(self.hidden_size * 2,
                                          self.mixture_size, bias=True)  # * 2 for bidirectional
        self.init_hidden(minibatch_size)
        self._dehidrals = Dihedral(num_vocab, alphabet_size, minibatch_size)
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

    def _get_network_emissions(self, original_aa_string, pssm, token):
        packed_input_sequences = self.embed(original_aa_string, pssm)
        minibatch_size = int(packed_input_sequences[1][0])
        self.init_hidden(minibatch_size)

        (data, bi_lstm_batches, _, _), self.hidden_layer = self.bi_lstm(packed_input_sequences, self.hidden_layer)
        emissions_padded, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(torch.nn.utils.rnn.PackedSequence(self.hidden_to_labels(data), bi_lstm_batches))
        emissions = emissions_padded.transpose(0, 1).transpose(1, 2)  # minibatch_size, self.mixture_size, -1
        emissions = self.batch_norm(emissions)
        emissions = emissions.transpose(1, 2).transpose(0, 1) # (minibatch_size, -1, self.mixture_size)

        output_angles = self._dehidrals(emissions, self.alphabet)
        backbone_atoms_padded, _ = \
            get_backbone_positions_from_angular_prediction(output_angles,
                                                           batch_sizes,
                                                           self.use_gpu)
        return output_angles, backbone_atoms_padded, batch_sizes

