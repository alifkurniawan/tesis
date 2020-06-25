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


class UTGN(openprotein.BaseModel):
    def __init__(self, input_dim=42, num_vocab=256, embedding_size=42, use_gpu=False):
        super().__init__(use_gpu, embedding_size)
        self.input_dim = input_dim
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_size
        self.use_gpu = use_gpu
        self.num_vocab = num_vocab

        self.emb = ProteinBertModel.from_pretrained('bert-base')

        self.W = nn.Linear(self.embedding_dim, self.num_vocab)

        # Share the weight matrix between target word embedding & the final logit dense layer
        # self.W.weight = self.emb.weight

        self.softmax = nn.Softmax(dim=1)

        ## POSITIONAL MASK
        self.mask = nn.Parameter(I.constant_(torch.empty(11, self.embedding_dim), 1));


    def _get_network_emissions(self, original_aa_string, pssm):
        # set input
        packed_input_sequences = self.embed(original_aa_string)
        minibatch_size = int(packed_input_sequences[1][0])

        return self._encoder_model(original_aa_string)
