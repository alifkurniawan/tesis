"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""
import math
import time
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
from util import calculate_dihedral_angles_over_minibatch, calc_angular_difference, \
    write_out, calc_rmsd, calc_drmsd, calculate_dihedral_angles, \
    get_structure_from_angles, write_to_pdb, calc_avg_drmsd_over_minibatch
from tape import ProteinBertModel


class BaseModel(nn.Module):
    def __init__(self, use_gpu, embedding_size, pretraining='bert-base'):
        super(BaseModel, self).__init__()

        # initialize model variables
        self.use_gpu = use_gpu
        self.historical_rmsd_avg_values = list()
        self.historical_drmsd_avg_values = list()

        if pretraining == 'bert-base':
            self.emb = ProteinBertModel.from_pretrained(pretraining)
            self.embedding_size = 768

    def get_embedding_size(self):
        return self.embedding_size

    def embed(self, original_aa_string, pssm, primary_token):
        start_compute_embed = time.time()

        if primary_token != -1 and original_aa_string is -1:
            tokens, token_batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
                torch.nn.utils.rnn.pack_sequence(primary_token))

            embeddings = torch.zeros(len(token_batch_sizes), token_batch_sizes[0], self.embedding_size)
            tokens = tokens.transpose(0, 1)

            for idx in range(len(tokens)):
                i = torch.tensor([tokens[idx].numpy()], dtype=torch.long)
                if self.use_gpu:
                    i = i.cuda()

                embeddings[idx] = self.emb(i)[0][0]

            embeddings = embeddings.transpose(0, 1)

            packed_input_sequences = rnn_utils.pack_padded_sequence(embeddings, token_batch_sizes)
        elif original_aa_string is not -1 and primary_token is not -1:

            tokens, token_batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
                torch.nn.utils.rnn.pack_sequence(primary_token))

            embeddings = torch.zeros(len(token_batch_sizes), token_batch_sizes[0], self.embedding_size - 21)
            tokens = tokens.transpose(0, 1)

            for idx in range(len(tokens)):
                i = torch.tensor([tokens[idx].numpy()], dtype=torch.long)
                if self.use_gpu:
                    i = i.cuda()

                embeddings[idx] = self.emb(i)[0][0]

            embeddings = embeddings.transpose(0, 1)

            data, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
                torch.nn.utils.rnn.pack_sequence(original_aa_string))

            # one-hot encoding
            prot_aa_list = data.unsqueeze(1)
            embed_tensor = torch.zeros(prot_aa_list.size(0), 21, prot_aa_list.size(2))  # 21 classes
            if self.use_gpu:
                prot_aa_list = prot_aa_list.cuda()
                embed_tensor = embed_tensor.cuda()
            one_hot_encoding = embed_tensor.scatter_(1, prot_aa_list.data, 1).transpose(1, 2)

            input_sequences = torch.zeros(one_hot_encoding.size(0), one_hot_encoding.size(1),
                                          one_hot_encoding.size(2) + embeddings.size(2))
            input_sequences[:, :, :one_hot_encoding.size(2)] = one_hot_encoding
            input_sequences[:, :, one_hot_encoding.size(2):] = embeddings

            packed_input_sequences = rnn_utils.pack_padded_sequence(input_sequences, batch_sizes)

        else:
            data, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
                torch.nn.utils.rnn.pack_sequence(original_aa_string))

            # one-hot encoding
            prot_aa_list = data.unsqueeze(1)
            embed_tensor = torch.zeros(prot_aa_list.size(0), 21, prot_aa_list.size(2))  # 21 classes
            if self.use_gpu:
                prot_aa_list = prot_aa_list.cuda()
                embed_tensor = embed_tensor.cuda()
            one_hot_encoding = embed_tensor.scatter_(1, prot_aa_list.data, 1).transpose(1, 2)

            # add pssm as input
            if pssm is not -1:
                pssm_data, pssm_batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
                    torch.nn.utils.rnn.pack_sequence(pssm))

                input_sequences = torch.zeros(one_hot_encoding.size(0), one_hot_encoding.size(1),
                                              one_hot_encoding.size(2) + pssm_data.size(2))
                input_sequences[:, :, :one_hot_encoding.size(2)] = one_hot_encoding
                input_sequences[:, :, pssm_data.size(2):] = pssm_data
            else:
                input_sequences = one_hot_encoding
            packed_input_sequences = rnn_utils.pack_padded_sequence(input_sequences, batch_sizes)

        end = time.time()
        write_out("Embed time:", end - start_compute_embed)
        return packed_input_sequences

    def compute_loss(self, minibatch):
        (original_aa_string, actual_coords_list, _, pssms, token) = minibatch

        emissions, _backbone_atoms_padded, _batch_sizes = self._get_network_emissions(original_aa_string, pssms, token)
        actual_coords_list_padded, batch_sizes_coords = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(actual_coords_list))
        if self.use_gpu:
            actual_coords_list_padded = actual_coords_list_padded.cuda()
        start = time.time()
        emissions_actual, _ = calculate_dihedral_angles_over_minibatch(actual_coords_list_padded,
                                                                       batch_sizes_coords,
                                                                       self.use_gpu)
        drmsd_avg = calc_avg_drmsd_over_minibatch(_backbone_atoms_padded,
                                                  actual_coords_list_padded,
                                                  _batch_sizes)
        write_out("Angle calculation time:", time.time() - start)
        if self.use_gpu:
            emissions_actual = emissions_actual.cuda()
            drmsd_avg = drmsd_avg.cuda()
        angular_loss = calc_angular_difference(emissions, emissions_actual)

        return angular_loss, drmsd_avg

    def forward(self, original_aa_string, pssm=-1, token=-1):
        return self._get_network_emissions(original_aa_string, pssm, token)

    def evaluate_model(self, data_loader):
        loss = 0
        angular_loss = 0
        data_total = []
        dRMSD_list = []
        RMSD_list = []
        for _, data in enumerate(data_loader, 0):
            primary_sequence, tertiary_positions, _mask, pssm, token = data
            start = time.time()
            predicted_angles, backbone_atoms, _batch_sizes = self(primary_sequence, pssm, token)
            write_out("Apply model to validation minibatch:", time.time() - start)
            cpu_predicted_angles = predicted_angles.transpose(0, 1).cpu().detach()
            cpu_predicted_backbone_atoms = backbone_atoms.transpose(0, 1).cpu().detach()
            minibatch_data = list(zip(primary_sequence,
                                      tertiary_positions,
                                      cpu_predicted_angles,
                                      cpu_predicted_backbone_atoms))
            data_total.extend(minibatch_data)
            actual_coords_list_padded, batch_sizes_coords = torch.nn.utils.rnn.pad_packed_sequence(
                torch.nn.utils.rnn.pack_sequence(tertiary_positions))
            if self.use_gpu:
                actual_coords_list_padded = actual_coords_list_padded.cuda()
            emissions_actual, _ = calculate_dihedral_angles_over_minibatch(actual_coords_list_padded,
                                                                           batch_sizes_coords,
                                                                           self.use_gpu)

            start = time.time()
            for primary_sequence, tertiary_positions, _predicted_pos, predicted_backbone_atoms \
                    in minibatch_data:
                actual_coords = tertiary_positions.transpose(0, 1).contiguous().view(-1, 3)
                predicted_coords = predicted_backbone_atoms[:len(primary_sequence)] \
                    .transpose(0, 1).contiguous().view(-1, 3).detach()

                if self.use_gpu:
                    emissions_actual = emissions_actual.cuda()
                angular_loss += float(calc_angular_difference(predicted_angles, emissions_actual))

                rmsd = calc_rmsd(predicted_coords, actual_coords)
                drmsd = calc_drmsd(predicted_coords, actual_coords)
                RMSD_list.append(rmsd)
                dRMSD_list.append(drmsd)
                error = float(drmsd)
                loss += error
                end = time.time()
            write_out("Calculate validation loss for minibatch took:", end - start)
        loss /= data_loader.dataset.__len__()
        angular_loss /= data_loader.dataset.__len__()
        self.historical_rmsd_avg_values.append(float(torch.Tensor(RMSD_list).mean()))
        self.historical_drmsd_avg_values.append(float(torch.Tensor(dRMSD_list).mean()))

        prim = data_total[0][0]
        pos = data_total[0][1]
        pos_pred = data_total[0][3]
        if self.use_gpu:
            pos = pos.cuda()
            pos_pred = pos_pred.cuda()
        angles = calculate_dihedral_angles(pos, self.use_gpu)
        angles_pred = calculate_dihedral_angles(pos_pred, self.use_gpu)

        write_to_pdb(get_structure_from_angles(prim, angles), "test")
        write_to_pdb(get_structure_from_angles(prim, angles_pred), "test_pred")

        data = {}
        data["pdb_data_pred"] = open("output/protein_test_pred.pdb", "r").read()
        data["pdb_data_true"] = open("output/protein_test.pdb", "r").read()
        data["phi_actual"] = list([math.degrees(float(v)) for v in angles[1:, 1]])
        data["psi_actual"] = list([math.degrees(float(v)) for v in angles[:-1, 2]])
        data["phi_predicted"] = list([math.degrees(float(v)) for v in angles_pred[1:, 1]])
        data["psi_predicted"] = list([math.degrees(float(v)) for v in angles_pred[:-1, 2]])
        data["rmsd_avg"] = self.historical_rmsd_avg_values
        data["drmsd_avg"] = self.historical_drmsd_avg_values

        prediction_data = None

        return loss, data, prediction_data, angular_loss
