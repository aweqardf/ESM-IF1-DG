# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import torch
import os
import esm

from numpy.ma.core import less_equal
from tqdm import tqdm
import numpy as np
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from esm.inverse_folding.classification_head import ClassificationHead, RegressionHead
from torch.autograd import Variable
from scipy.spatial import transform
from torch.utils.hipify.hipify_python import value

from esm.data import Alphabet

from .features import DihedralFeatures
from .gvp_encoder import GVPEncoder
from .gvp_utils import unflatten_graph
from .gvp_transformer_encoder import GVPTransformerEncoder
from .transformer_decoder import TransformerDecoder
from .util import rotate, CoordBatchConverter, hpatch_fast, extract_adjacency, hydrophobic_loss


SMALL_CONST = 1e-15
EPSILON = 1e-10
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)



class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a classifier head"""
    def __init__(
            self,
            class_size=None,
            discriminator_type='classification',
            output_type='classification',
            classifier_head=None,
            device='cuda'
    ):
        super().__init__()
        self.discriminator_type = discriminator_type
        self.output_type=output_type
        self.encoder, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.encoder = self.encoder.eval()
        self.hidden_dim = self.encoder.args.decoder_embed_dim
        self.device = device

        if classifier_head:
            self.classifier_head = classifier_head
        else:
            if self.discriminator_type == 'classification':
                if not class_size:
                    raise ValueError("must specify class_size")
                self.classifier_head = ClassificationHead(
                    class_size=class_size,
                    embed_size=self.hidden_dim
                )
            elif self.discriminator_type == 'regression':
                self.classifier_head = RegressionHead(
                    embed_size=self.hidden_dim
                )
            else:
                raise ValueError("discriminator_type must in 'classification' or 'regression'")

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x, coords):
        mask = x.ne(0).unsqueeze(2).repeat(
            1,1,self.hidden_dim
        ).float().to(self.device).detach()
        logits = self.encoder.get_hidden(
            coords,
            x,
            device=self.device
        )
        avg_hidden = torch.sum(logits, dim=2) / (
            torch.sum(mask, dim=1) + EPSILON
        )
        return avg_hidden

    def forward(self, x, coords):
        avg_hidden = self.avg_representation(x.to(self.device), coords)
        if self.discriminator_type == 'classification':
            result = self.classifier_head(avg_hidden)
            # result = F.log_softmax(logits, dim=-1)
        else:
            result = self.classifier_head(avg_hidden)
            if self.output_type=='classification':
                result=torch.sigmoid(result)
        return result

class GVPTransformerModel(nn.Module):
    """
    GVP-Transformer inverse folding model.

    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """

    def __init__(self, args, alphabet):
        super().__init__()
        encoder_embed_tokens = self.build_embedding(
            args, alphabet, args.encoder_embed_dim,
        )
        decoder_embed_tokens = self.build_embedding(
            args, alphabet, args.decoder_embed_dim, 
        )
        encoder = self.build_encoder(args, alphabet, encoder_embed_tokens)
        decoder = self.build_decoder(args, alphabet, decoder_embed_tokens)
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = GVPTransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
        )
        return decoder

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.padding_idx
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward(
        self,
        coords,
        padding_mask,
        confidence,
        prev_output_tokens,
        return_all_hiddens: bool = False,
        features_only: bool = False,
    ):
        encoder_out = self.encoder(coords, padding_mask, confidence,
            return_all_hiddens=return_all_hiddens)
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra

    def forward_t(
            self,
            coords,
            seq,
            temperature=1.0,
            confidence=None,
            features_only=False,
            device=None
    ):
        L = len(coords)
        # Convert to batch format
        batch_converter = CoordBatchConverter(self.decoder.dictionary)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coords, confidence, None)], device=device)
        )
        # batch_coords, confidence, _, _, padding_mask = (
        #     batch_converter(coords, device=device)
        # )

        encoder_out = self.encoder(batch_coords, padding_mask, confidence)
        logits, extra = self.decoder(
            seq,
            encoder_out=encoder_out,
            features_only=features_only,
        )
        logits = logits[0].transpose(0, 1)
        logits = logits[-1:]
        logits /= temperature
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_hidden(
            self,
            coords,
            seq,
            confidence=None,
            features_only=True,
            device=None
    ):
        L = len(coords)
        # Convert to batch format
        batch_converter = CoordBatchConverter(self.decoder.dictionary)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter(coords, device=device)
        )
        # batch_coords, confidence, _, _, padding_mask = (
        #     batch_converter([(coords, confidence, None)], device=device)
        # )

        encoder_out = self.encoder(batch_coords, padding_mask, confidence)
        hidden, _ = self.decoder(
            seq,
            encoder_out=encoder_out,
            features_only=features_only,
        )
        return hidden
    
    def sample(self, coords, partial_seq=None, temperature=1.0, confidence=None, device=None):
        """
        Samples sequences based on multinomial sampling (no beam search).

        Args:
            coords: L x 3 x 3 list representing one backbone
            partial_seq: Optional, partial sequence with mask tokens if part of
                the sequence is known
            temperature: sampling temperature, use low temperature for higher
                sequence recovery and high temperature for higher diversity
            confidence: optional length L list of confidence scores for coordinates
        """
        L = len(coords)
        # Convert to batch format
        batch_converter = CoordBatchConverter(self.decoder.dictionary)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coords, confidence, None)], device=device)
        )
        
        # Start with prepend token
        mask_idx = self.decoder.dictionary.get_idx('<mask>')
        sampled_tokens = torch.full((1, 1+L), mask_idx, dtype=int)
        sampled_tokens[0, 0] = self.decoder.dictionary.get_idx('<cath>')
        if partial_seq is not None:
            for i, c in enumerate(partial_seq):
                sampled_tokens[0, i+1] = self.decoder.dictionary.get_idx(c)
            
        # Save incremental states for faster sampling
        incremental_state = dict()
        
        # Run encoder only once
        encoder_out = self.encoder(batch_coords, padding_mask, confidence)
        
        # Make sure all tensors are on the same device if a GPU is present
        if device:
            sampled_tokens = sampled_tokens.to(device)
        
        # Decode one token at a time
        for i in range(1, L+1):
            logits, _ = self.decoder(
                sampled_tokens[:, :i], 
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            if sampled_tokens[0, i] == mask_idx:
                sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
        sampled_seq = sampled_tokens[0, 1:]
        
        # Convert back to string via lookup
        return ''.join([self.decoder.dictionary.get_tok(a) for a in sampled_seq])

    def sample_PPLM(self, coords, partial_seq=None, temperature=1.0, confidence=None,stepsize=0.5,num_iterarions=2, gamma=1.5, kl_scale=0.5, loss_type=1, device=None, position=-1):
        """
        Samples sequences based on multinomial sampling (no beam search).

        Args:
            coords: L x 3 x 3 list representing one backbone
            partial_seq: Optional, partial sequence with mask tokens if part of
                the sequence is known
            temperature: sampling temperature, use low temperature for higher
                sequence recovery and high temperature for higher diversity
            confidence: optional length L list of confidence scores for coordinates
        """
        L = len(coords)
        extra_data = {}
        # Convert to batch format (converter need list(tuple1,tuple2,...) format)
        batch_converter = CoordBatchConverter(self.decoder.dictionary)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coords, confidence, None)], device=device)
        )

        # Start with prepend token
        mask_idx = self.decoder.dictionary.get_idx('<mask>')
        sampled_tokens = torch.full((1, 1 + L), mask_idx, dtype=int)
        sampled_tokens[0, 0] = self.decoder.dictionary.get_idx('<cath>')
        if partial_seq is not None:
            for i, c in enumerate(partial_seq):
                sampled_tokens[0, i + 1] = self.decoder.dictionary.get_idx(c)

        # Save incremental states for faster sampling
        incremental_state = dict()
        init_incremental_state = dict()

        # Run encoder only once
        with torch.no_grad():
            encoder_out = self.encoder(batch_coords, padding_mask, confidence)

        # Make sure all tensors are on the same device if a GPU is present
        if device:
            sampled_tokens = sampled_tokens.to(device)

        classifier_thermo = get_classifier(class_size=2,
                                    checkpoint_path=f'{current_dir}/discriminator_param/classifier_head_stability_epoch_last.pt',
                                    discriminator_type='regression',
                                    output_type='classification')
        classifier_thermo.to('cuda')
        classifier_thermo.eval()

        classifier_solu = get_classifier(class_size=2,
                                           checkpoint_path=f'{current_dir}/discriminator_param/classifier_head_solubility_epoch_last.pt',
                                           discriminator_type='classification',
                                           output_type='classification')
        classifier_solu.to('cuda')
        classifier_solu.eval()
        classifier_dict = {
            'thermo':classifier_thermo,
            'solu':classifier_solu
        }




        # Decode one token at a time
        for i in tqdm(range(1, L + 1)):

            with torch.no_grad():
                unpert_logits, _ = self.decoder(
                    sampled_tokens[:, :i],
                    encoder_out,
                    incremental_state=init_incremental_state,
                )

            if incremental_state is not None and len(incremental_state) != 0:
                pert_incremental_state = perturb_past(
                    sampled_tokens,
                    encoder_out,
                    unpert_logits,
                    incremental_state,
                    self.decoder,
                    classifier_dict,
                    stepsize=stepsize,
                    num_iterations=num_iterarions,
                    gamma=gamma,
                    loss_type=loss_type,
                    kl_scale=kl_scale,
                    position=i-1,
                )
            else:
                pert_incremental_state = incremental_state

            with torch.no_grad():
                logits, _ = self.decoder(
                    sampled_tokens[:, :i],
                    encoder_out,
                    incremental_state=pert_incremental_state,
                )
            incremental_state = pert_incremental_state
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            if sampled_tokens[0, i] == mask_idx:
                sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
            if i == position:
                extra_data[i] = probs
                break
        sampled_seq = sampled_tokens[0, 1:]

        # Convert back to string via lookup
        return ''.join([self.decoder.dictionary.get_tok(a) for a in sampled_seq]), extra_data


def to_var(d, indices_to_convert=[0,2,4,6,8,10,12,14]):
    result = {}
    for i, (key, value) in enumerate(d.items()):
        if i in indices_to_convert:
            prev_key = value['prev_key']
            prev_value = value['prev_value']
            prev_key_padding_mask = value['prev_key_padding_mask']
            result[key] = {
                'prev_key': prev_key.clone().detach().requires_grad_(True),
                'prev_value': prev_value.clone().detach().requires_grad_(True),
                'prev_key_padding_mask': prev_key_padding_mask
            }
        else:
            result[key] = value
    return result

def norm_grad(d, stepsize, gamma=1.0, indices_to_convert=[0,2,4,6,8,10,12,14]):
    grad = {}
    for i, (key, value) in enumerate(d.items()):
        if i in indices_to_convert:
            prev_key = value['prev_key'].grad
            prev_value = value['prev_value'].grad
            prev_key_padding_mask = value['prev_key_padding_mask']
            grad[key] = {
                'prev_key': -stepsize * prev_key/(torch.norm(prev_key)+SMALL_CONST)**gamma,
                'prev_value': -stepsize * prev_value/torch.norm(prev_value+SMALL_CONST)**gamma,
                'prev_key_padding_mask': prev_key_padding_mask
            }
        else:
            grad[key] = {
                'prev_key': torch.zeros_like(value['prev_key']),
                'prev_value': torch.zeros_like(value['prev_value']),
                'prev_key_padding_mask': torch.zeros_like(value['prev_key_padding_mask'])
            }
    return grad



def dict_clone(d):
    result = {}
    for i, (key, value) in enumerate(d.items()):
        prev_key = value['prev_key']
        prev_value = value['prev_value']
        prev_key_padding_mask = value['prev_key_padding_mask']
        result[key] = {
            'prev_key': prev_key.clone().detach().requires_grad_(False),
            'prev_value': prev_value.clone().detach().requires_grad_(False),
            'prev_key_padding_mask': prev_key_padding_mask
        }
    return result

def dict_detach(d):
    result = {}



def add_tensors(a, b):
    result = {}
    for layer_id, layer_dict in a.items():
        # 确保 result 初始化了 layer_id 的子字典
        result[layer_id] = {}
        for key, value in layer_dict.items():
            inc_value = b[layer_id][key]
            if value is not None:
                result[layer_id][key] = value + inc_value
            else:
                # 如果 value 为 None，则保持 None
                result[layer_id][key] = None
    return result

def get_classifier(class_size=2,
                   checkpoint_path=f'{current_dir}/discriminator_param/classifier_head_solubility_epoch_last.pt',
                   discriminator_type='regression',
                   output_type='classification'):
    discriminator = Discriminator(class_size=class_size,discriminator_type=discriminator_type,output_type=output_type)
    discriminator.get_classifier().load_state_dict(torch.load(checkpoint_path))
    for param in discriminator.encoder.parameters():
        param.requires_grad = False
    discriminator.classifier_head.requires_grad = False

    return discriminator

def perturb_past(
        sampled_tokens,
        encoder_out,
        unpert_logits,
        incremental_state,
        model,
        classifier_dict,
        position,
        class_label=1,
        stepsize=0.5,
        num_iterations = 2,
        gamma=1.0,
        kl_scale = 0.5,
        loss_type = 2,
        device='cuda'
):
    grad_accumulator = {
        layer_id: {
            key: (
                torch.zeros(value.shape, dtype=torch.float32, device=device)
                if hasattr(value, 'shape') else None
            )
            for key, value in layer_dict.items()
        }
        for layer_id, layer_dict in incremental_state.items()
    }

    curr_incremental_state = dict_clone(incremental_state)
    hidden, _ = model.extract_features(
        sampled_tokens,
        encoder_out,
        incremental_state=curr_incremental_state,
    )
    avg_hidden = torch.mean(hidden, dim=1)
    avg_hidden = avg_hidden.detach()

    unpert_probs = F.softmax(unpert_logits, dim=1)



    for i in range(num_iterations):
        curr_perturb = to_var(grad_accumulator)
        curr_incremental_state = dict_clone(incremental_state)

        perturbed_incremental_state = add_tensors(curr_incremental_state, curr_perturb)
        hidden_pert, _ = model.extract_features(
            sampled_tokens,
            encoder_out,
            incremental_state=perturbed_incremental_state,
        )
        avg_hidden_pert = torch.mean(hidden_pert,dim=1)
        logits = model.output_layer(hidden_pert)
        logits = logits.transpose(1,2)
        probs = F.softmax(logits,dim=1)

        loss = 0.0

        if loss_type == 1:
            bce_loss = torch.nn.BCEWithLogitsLoss()
            prediction_class = classifier_dict['solu'].classifier_head(avg_hidden_pert)
            label = torch.ones_like(prediction_class, device=device) * class_label
            discrim_loss = bce_loss(prediction_class, label)
            loss += discrim_loss


        if loss_type == 2:
            bce_loss = torch.nn.BCEWithLogitsLoss()
            prediction = classifier_dict['thermo'].classifier_head(avg_hidden_pert - avg_hidden)
            prediction_class = torch.sigmoid(prediction)
            label = torch.ones_like(prediction_class, device=device) * class_label
            discrim_loss = bce_loss(prediction_class, label)
            loss += discrim_loss

        if loss_type == 3:
            bce_loss = torch.nn.BCEWithLogitsLoss()
            prediction_class = classifier_dict['solu'].classifier_head(avg_hidden_pert)
            label = torch.ones_like(prediction_class, device=device) * class_label
            discrim_loss_solu = bce_loss(prediction_class, label)
            loss += discrim_loss_solu *3

            prediction_thermo = classifier_dict['thermo'].classifier_head(avg_hidden_pert - avg_hidden)
            prediction_class_thermo = torch.sigmoid(prediction_thermo)
            label = torch.ones_like(prediction_class_thermo, device=device) * class_label
            discrim_loss_thermo = bce_loss(prediction_class_thermo, label)

            loss += discrim_loss_thermo




        kl_loss = 0.0
        if kl_scale > 0:
            kl_loss += kl_scale * (probs[0,:,0]*(probs[0,:,0]/unpert_probs[0,:,0]).log()).sum()
            loss += kl_loss

        loss.backward()

        # normalize gradients
        grad = norm_grad(curr_perturb, stepsize,gamma=gamma)

        # accumulate grad
        grad_accumulator = add_tensors(grad, grad_accumulator)

    pert_incremental_state = add_tensors(incremental_state, grad_accumulator)

    return pert_incremental_state
