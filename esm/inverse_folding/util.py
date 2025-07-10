# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math

import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
import numpy as np
from scipy.spatial import transform
from scipy.stats import special_ortho_group
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from typing import Sequence, Tuple, List

from esm.data import BatchConverter

import pandas as pd
import bisect
from Bio.PDB import PDBParser, NeighborSearch
from esm.data import Alphabet


def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('pdb'):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return coords, seq


def load_coords(fpath, chain):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    structure = load_structure(fpath, chain)
    return extract_coords_from_structure(structure)


def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """
    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)


def get_sequence_loss(model, alphabet, coords, seq):
    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, seq)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)

    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx)
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
    loss = F.cross_entropy(logits, target, reduction='none')
    loss = loss[0].cpu().detach().numpy()
    target_padding_mask = target_padding_mask[0].cpu().numpy()
    return loss, target_padding_mask


def score_sequence(model, alphabet, coords, seq):
    loss, target_padding_mask = get_sequence_loss(model, alphabet, coords, seq)
    ll_fullseq = -np.sum(loss * ~target_padding_mask) / np.sum(~target_padding_mask)
    # Also calculate average when excluding masked portions
    coord_mask = np.all(np.isfinite(coords), axis=(-1, -2))
    ll_withcoord = -np.sum(loss * coord_mask) / np.sum(coord_mask)
    return ll_fullseq, ll_withcoord


def get_encoder_output(model, alphabet, coords):
    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, None)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)
    encoder_out = model.encoder.forward(coords, padding_mask, confidence,
            return_all_hiddens=False)
    # remove beginning and end (bos and eos tokens)
    return encoder_out['encoder_out'][0][1:-1, 0]


def rotate(v, R):
    """
    Rotates a vector by a rotation matrix.
    
    Args:
        v: 3D vector, tensor of shape (length x batch_size x channels x 3)
        R: rotation matrix, tensor of shape (length x batch_size x 3 x 3)

    Returns:
        Rotated version of v by rotation matrix R.
    """
    R = R.unsqueeze(-3)
    v = v.unsqueeze(-1)
    return torch.sum(v * R, dim=-2)


def get_rotation_frames(coords):
    """
    Returns a local rotation frame defined by N, CA, C positions.

    Args:
        coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
        where the third dimension is in order of N, CA, C

    Returns:
        Local relative rotation frames in shape (batch_size x length x 3 x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)
    return R


def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)


def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    v_expand = torch.unsqueeze(values, -1)
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
            torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )


class CoordBatchConverter(BatchConverter):
    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x 3 x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>") 
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))

        coords_and_confidence, strs, tokens = super().__call__(batch)

        # pad beginning and end of each protein due to legacy reasons
        coords = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.)
            for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
        padding_mask = torch.isnan(coords[:,:,0,0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.) * padding_mask
        return coords, confidence, strs, tokens, padding_mask

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result

##### hydrophobic patch #####

sasa_data = {
    'amino_acid': ['F', 'W', 'M', 'L', 'Y', 'I', 'K', 'V', 'P', 'H',
                   'C', 'R', 'T', 'A', 'E', 'Q', 'S', 'D', 'G', 'N'],
    'nbs1-10': [140.33, 97.01, 139.78, 95.72, 129.12, 97.25, 119.66, 85.25, 114.72, 83.56,
                114.64, 80.00, 101.83, 81.16, 97.18, 70.04, 93.59, 72.40, 90.26, 68.55],
    'nbs11-13': [80.59, 53.46, 71.21, 56.79, 68.93, 53.70, 63.72, 50.20, 59.10, 44.45,
                 51.72, 39.30, 47.41, 37.45, 42.50, 31.62, 38.24, 31.12, 36.32, 27.02],
    'nbs14-16': [58.46, 30.86, 58.85, 34.59, 61.79, 33.03, 53.91, 26.87, 55.49, 30.61,
                 52.77, 25.21, 63.23, 42.99, 47.46, 23.49, 52.45, 31.07, 48.19, 30.81],
    'nbs17-20': [32.72, 16.52, 42.58, 26.33, 38.04, 21.79, 35.41, 17.71, 32.77, 19.46,
                 29.56, 17.62, 27.29, 14.82, 22.33, 12.67, 23.67, 13.42, 19.65, 10.92],
    'nbs21-24': [11.48, 16.70, 11.05, 7.93, 13.05, 7.48, 23.74, 6.79, 12.69, 15.02,
                 5.64, 11.48, 8.33, 5.48, 8.16, 7.27, 5.37, 5.78, 5.24, 4.51],
    'nbs24': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0]
}

nll_data = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.443, 0.443, 0.443, 0.443, 0.443, 0.127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.706, 0.706, 0.706, 0.706, 0.706, 0.138, 0.076, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.929, 0.929, 0.929, 0.929, 0.929, 0.348, 0.182, 0.113, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1.099, 1.099, 1.099, 1.099, 1.099, 0.849, 0.379, 0.124, 0.023, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1.686, 1.686, 1.686, 1.686, 1.686, 0.937, 0.579, 0.418, 0.243, 0.017, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2.315, 2.315, 2.315, 2.315, 2.315, 1.376, 1.084, 0.789, 0.534, 0.169, 0.174, 0.080, 0.009, 0, 0, 0, 0, 0, 0, 0],
    [2.092, 2.092, 2.092, 2.092, 2.092, 1.713, 1.372, 1.079, 0.628, 0.472, 0.320, 0.145, 0.066, 0.063, 0, 0, 0, 0, 0,
     0],
    [2.785, 2.785, 2.785, 2.785, 2.785, 2.378, 1.883, 1.340, 1.160, 0.732, 0.617, 0.415, 0.118, 0.184, 0.039, 0, 0,
     0.008, 0, 0.053],
    [4.394, 4.394, 4.394, 4.394, 4.394, 2.133, 2.261, 1.960, 1.470, 1.055, 0.910, 0.571, 0.362, 0.330, 0.165, 0.100,
     0.071, 0.018, 0.005, 0.049],
    [25, 25, 25, 25, 25, 3.071, 2.346, 2.072, 1.769, 1.359, 1.211, 0.897, 0.595, 0.580, 0.295, 0.216, 0.147, 0.098,
     0.101, 0.035],
    [25, 25, 25, 25, 25, 3.476, 3.387, 2.531, 2.103, 1.741, 1.660, 1.154, 0.911, 0.815, 0.466, 0.354, 0.279, 0.152,
     0.149, 0.087],
    [25, 25, 25, 25, 25, 3.659, 3.569, 3.070, 2.430, 2.302, 1.912, 1.493, 1.171, 1.032, 0.836, 0.539, 0.532, 0.328,
     0.344, 0.250],
    [25, 25, 25, 25, 25, 25, 4.773, 3.267, 2.920, 2.552, 2.503, 2.038, 1.572, 1.443, 1.039, 0.725, 0.694, 0.490, 0.515,
     0.322],
    [25, 25, 25, 25, 25, 25, 4.262, 4.323, 3.201, 2.918, 2.963, 2.204, 1.828, 1.803, 1.370, 0.987, 0.902, 0.796, 0.646,
     0.579],
    [25, 25, 25, 25, 25, 25, 5.872, 4.323, 4.414, 3.059, 3.196, 2.747, 2.297, 2.126, 1.789, 1.340, 1.161, 0.961, 0.831,
     0.766],
    [25, 25, 25, 25, 25, 25, 5.872, 4.205, 3.922, 3.697, 3.433, 3.004, 2.591, 2.640, 2.046, 1.574, 1.610, 1.375, 1.004,
     0.985],
    [25, 25, 25, 25, 25, 25, 5.872, 5.016, 3.979, 3.851, 3.743, 3.633, 2.947, 3.209, 2.482, 1.927, 1.903, 1.469, 1.407,
     1.284],
    [25, 25, 25, 25, 25, 25, 25, 5.016, 4.733, 3.894, 3.907, 3.989, 3.242, 3.163, 2.670, 2.271, 2.343, 1.949, 1.825,
     1.550],
    [25, 25, 25, 25, 25, 25, 25, 6.402, 5.203, 4.464, 4.467, 4.549, 4.057, 3.812, 3.309, 2.766, 2.687, 2.175, 1.953,
     1.780],
    [25, 25, 25, 25, 25, 25, 25, 6.402, 6.812, 4.390, 4.675, 4.488, 4.357, 4.323, 3.670, 3.039, 2.978, 2.532, 2.426,
     2.287],
    [25, 25, 25, 25, 25, 25, 25, 6.402, 5.714, 4.390, 5.042, 4.924, 4.955, 4.534, 4.156, 3.504, 3.686, 3.092, 2.914,
     2.498],
    [25, 25, 25, 25, 25, 25, 25, 6.402, 6.812, 4.390, 5.294, 5.935, 5.156, 5.170, 4.695, 3.921, 4.106, 3.425, 3.068,
     2.849],
    [25, 25, 25, 25, 25, 25, 25, 25, 6.812, 5.643, 5.448, 5.935, 5.407, 5.576, 4.338, 4.343, 4.063, 3.720, 3.401,
     3.171],
    [25, 25, 25, 25, 25, 25, 25, 25, 6.812, 7.029, 6.546, 5.530, 5.561, 5.758, 5.137, 5.091, 4.938, 4.788, 4.027,
     3.787],
    [25, 25, 25, 25, 25, 25, 25, 25, 25, 7.029, 7.239, 6.628, 5.967, 5.981, 5.137, 5.678, 4.843, 4.883, 4.860, 4.193],
    [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 7.239, 6.628, 5.967, 6.674, 6.641, 5.496, 7.241, 6.087, 5.217, 4.685],
    [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 7.239, 6.628, 6.660, 6.674, 5.948, 5.902, 7.241, 6.493, 5.217, 4.781],
    [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 7.322, 6.660, 6.674, 5.948, 5.902, 7.241, 6.493, 6.064, 4.886],
    [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 7.368, 7.334, 6.595, 6.548, 6.493, 7.163, 6.390],
    [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 7.334, 6.595, 6.548, 6.493, 6.470, 25],
    [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 7.334, 7.288, 7.241, 6.493, 25, 25],
    [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
]

sasa_df = pd.DataFrame(sasa_data)
nll_data = torch.tensor(nll_data)


def extract_adjacency(pdb_file, chain_id='A', distance_cutoff=10.0):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    ca_atoms = []
    ca_atom_map = {}
    idx = 0
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if "CA" in residue:
                        ca_atoms.append(residue["CA"])
                        ca_atom_map[residue.id[1]] = idx
                        idx += 1

    ns = NeighborSearch(ca_atoms)

    seq_length = len(ca_atoms)
    residue_list = []
    nbs_list = []
    adjacency = np.zeros([seq_length, seq_length])

    for atom in ca_atoms:
        residue = atom.get_parent()
        residue_id = residue.id[1]
        residue_list.append(residue.get_resname())

        neighbors = ns.search(atom.coord, distance_cutoff)
        nbs_list.append(len(neighbors))
        for neighbor in neighbors:
            neighbor_id = neighbor.get_parent().id[1]
            adjacency[ca_atom_map[residue_id], ca_atom_map[neighbor_id]] = 1

    return residue_list, nbs_list, adjacency

def surface_mask(pdb_file, chain_id='A', nbs_cutoff=16,device='cuda'):
    _, nbs_list, _ = extract_adjacency(pdb_file,chain_id=chain_id)
    nbs_list = torch.tensor(nbs_list)
    mask = (nbs_list < nbs_cutoff).int()
    return mask



def hpatch_fast(pdb_file, seq_list, position):
    residue_list, nbs_list, adjacency = extract_adjacency(pdb_file)
    adjacency_mtx = torch.from_numpy(adjacency)

    dictionary = Alphabet.from_architecture('invariant_gvp')
    aa_list = sasa_df['amino_acid'].values
    indices = [dictionary.get_idx(aa) for aa in aa_list]
    indices = torch.tensor(indices)


    sasa_list = []
    nbs_ranges = [10, 13, 16, 20, 24]
    nbs_columns = ['nbs1-10', 'nbs11-13', 'nbs14-16', 'nbs17-20', 'nbs21-24', 'nbs24']
    for idx, seq in enumerate(seq_list[0,1:]):
        nbs_idx = bisect.bisect_right(nbs_ranges, nbs_list[idx])
        nbs_idx = nbs_columns[nbs_idx]
        if dictionary.get_tok(seq) not in aa_list:
            sasa_list.append(sasa_df[nbs_idx].values.mean())
        else:
            sasa_list.append(sasa_df[sasa_df['amino_acid'] == dictionary.get_tok(seq)][nbs_idx].values[0])
        if idx == position:
            sasa_current = torch.tensor(sasa_df[nbs_idx].values)

    sasa_mtx = torch.from_numpy(np.array(sasa_list)).unsqueeze(-1).expand(-1, 20).clone()
    sasa_mtx[position] = sasa_current
    sasa_all = torch.mm(adjacency_mtx, sasa_mtx)

    nbs_mtx = torch.tensor(nbs_list)
    nbs_mtx = nbs_mtx.unsqueeze(-1).expand(-1, 20)

    nbs_index = nbs_mtx.to(torch.long) - 1
    # sasa_index = (sasa_all // 25).to(torch.long)
    sasa_index = torch.div(sasa_all, 25, rounding_mode='trunc').to(torch.long)

    nll_mtx = torch.ones_like(sasa_index, dtype=nll_data.dtype) * 25
    valid_mask = (sasa_index >= 0) & (sasa_index < 45) & (nbs_index >= 0) & (nbs_index < 20)
    valid_sasa_index = sasa_index[valid_mask]
    valid_nbs_index = nbs_index[valid_mask]

    nll_mtx[valid_mask] = nll_data[valid_sasa_index, valid_nbs_index]

    nll_mtx_sum = nll_mtx[:position].sum(dim=0)
    result = torch.zeros(35)
    result.scatter_(0, indices, nll_mtx_sum)
    return result


def hpatch_fast_test(pdb_file, seq_list):
    residue_list, nbs_list, adjacency = extract_adjacency(pdb_file)
    adjacency_mtx = torch.from_numpy(adjacency)

    dictionary = Alphabet.from_architecture('invariant_gvp')
    aa_list = sasa_df['amino_acid'].values
    indices = [dictionary.get_idx(aa) for aa in aa_list]
    indices = torch.tensor(indices)


    sasa_list = []
    nbs_ranges = [10, 13, 16, 20, 24]
    nbs_columns = ['nbs1-10', 'nbs11-13', 'nbs14-16', 'nbs17-20', 'nbs21-24', 'nbs24']
    for idx, seq in enumerate(seq_list):
        nbs_idx = bisect.bisect_right(nbs_ranges, nbs_list[idx])
        nbs_idx = nbs_columns[nbs_idx]
        sasa_list.append(sasa_df[sasa_df['amino_acid'] == seq][nbs_idx].values[0])

    sasa_mtx = torch.from_numpy(np.array(sasa_list)).unsqueeze(-1).clone()
    sasa_all = torch.mm(adjacency_mtx, sasa_mtx).squeeze(-1).numpy()

    return sasa_all, nbs_list

##### hydrophobic set #####
def hydrophobic_loss(probs,device='cuda'):
    hydrophobic_indices = [[8,9,10,11,13,15,16,17,19,21]] #hydrophobic aa index
    hydrophobic_indices = torch.tensor(hydrophobic_indices).to(device)
    num_hydrophobic = hydrophobic_indices.shape[0]
    hydrophobic_onehot = torch.zeros(num_hydrophobic, 35).to(device)
    hydrophobic_onehot.scatter_(1,hydrophobic_indices,1)
    logits = torch.mm(probs, torch.t(hydrophobic_onehot))
    loss = -torch.log(torch.sum(logits))
    return loss


if __name__ == '__main__':
    dictionary = Alphabet.from_architecture('invariant_gvp')
    # hydrophobic = ['A', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W']
    hydrophilic = ['R', 'K', 'D', 'E', 'N', 'Q', 'S', 'T', 'Y', 'H']
    for item in hydrophilic:
        print(dictionary.get_idx(item))
    # print(surface_mask('../../PPLM_dataset/AlphaFold_model_PDBs/1PGA.pdb'))



