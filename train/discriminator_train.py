import argparse
import csv
import json
import time

import numpy as np
import os
import sys
sys.path.insert(0,'../../esm/')
import torch
import torch.utils.data as data
import torch.nn.functional as F
import esm
from esm.inverse_folding.classification_head import ClassificationHead, RegressionHead
from esm.data import Alphabet



EPSILON = 1e-10

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


class Dataset(data.Dataset):
    def __init__(self, X, y, pdb_list):
        self.X = X
        self.y = y
        self.pdb_list = pdb_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        data = {}
        data['X'] = self.X[index]
        data['y'] = self.y[index]
        coords, _ = esm.inverse_folding.util.load_coords(f'../../PPLM_dataset/AlphaFold_model_PDBs/{self.pdb_list[index]}.pdb', 'A')
        data['coords'] = (coords,None,None)
        return data


def collate_fn(data, discriminator_type):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(
            len(sequences),
            max(lengths)
        ).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]

        return padded_sequences, lengths

    item_info = {}
    item_info['coords'] = []
    item_info['X'] = []
    item_info['y'] = []

    for d in data:
        if d['X'].shape[0] == d['coords'][0].shape[0]:
            item_info['coords'].append(d['coords'])
            item_info['X'].append(d['X'])
            item_info['y'].append(d['y'])


    x_batch, _ = pad_sequences(item_info["X"])
    if discriminator_type == 'classification':
        y_batch = torch.tensor(item_info["y"], dtype=torch.long)
    else:
        y_batch = torch.tensor(item_info["y"], dtype=torch.float)

    return x_batch, y_batch, item_info['coords']


def train_epoch(
        data_loader, discriminator, optimizer,
        epoch=0, log_interval=10, device ='cuda'
):
    samples_so_far = 0
    discriminator.train_custom()

    for batch_idx, (input_t, target_t, coords_t) in enumerate(data_loader):
        input_t, target_t = input_t.to(device), target_t.to(device)
        coords_t = [(torch.from_numpy(t[0]).to(device),None,None) for t in coords_t]
        optimizer.zero_grad()

        output_t = discriminator(input_t, coords_t)
        if discriminator.discriminator_type == 'classification':
            loss = F.nll_loss(output_t, target_t)
        else:
            target_t = target_t.unsqueeze(1)
            loss = F.mse_loss(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(input_t)

        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch + 1} [{samples_so_far}/{len(data_loader.dataset)}\tLoss: {loss.item():.6f}]"
            )


def evaluate_performance(data_loader, discriminator, device='cuda'):
    discriminator.eval()
    test_loss = 0.
    correct = 0.
    num_data = 0
    with torch.no_grad():
        for input_t, target_t, coords_t in data_loader:
            input_t, target_t = input_t.to(device), target_t.to(device)
            coords_t = [(torch.from_numpy(t[0]).to(device), None, None) for t in coords_t]
            output_t = discriminator(input_t, coords_t)
            if discriminator.discriminator_type == 'classification':
                test_loss += F.nll_loss(output_t, target_t, reduction='sum').item()
                pred_t = output_t.argmax(dim=1, keepdim=True)
                correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()
            else:
                target_t = target_t.unsqueeze(1)
                test_loss += F.mse_loss(output_t, target_t, reduction='sum').item()
            num_data += input_t.shape[0]

    test_loss /= num_data
    accuracy = correct/num_data

    print(
        "Performance on test set: "
        f"Average loss: {test_loss:.4f}"
    )
    if discriminator.discriminator_type == 'classification':
        print(f'Accuracy: {accuracy*100.:.0f}%')

    return test_loss, accuracy


def train_discriminator(
        dataset,
        discriminator_type="regression",
        epochs=100,
        learning_rate=0.001,
        batch_size=512,
        log_interval=10,
        save_model=True,
        output_fp='./discriminator_param'
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dictionary = Alphabet.from_architecture('invariant_gvp')

    if dataset == 'Stability':
        if discriminator_type == 'classification':
            idx2class = ['unstable', 'stable']
            class2idx = {c: i for i, c in enumerate(idx2class)}

            discriminator = Discriminator(
                class_size=len(idx2class),
                device=device,
            ).to(device)

            x=[]
            y=[]
            pdb_list = []
            with open("../PPLM_dataset/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv") as f:
                csv_reader = csv.reader(f)
                next(csv_reader)
                data = []
                for i, line in enumerate(csv_reader):
                    if i < 100000:
                        try:
                            seq = torch.tensor(dictionary.encode(line[-10]))
                            ddG = float(line[-3]) > 0.
                            pdb = line[-8].split('.')[0].replace("|", ":")
                            x.append(seq)
                            y.append(ddG)
                            pdb_list.append(pdb)
                        except:
                            print(f"error evaluating Line {i} {line[-10]} {line[-3]}")

        else:

            discriminator = Discriminator(
                discriminator_type=discriminator_type,
                device=device,
                output_type='regression'
            ).to(device)

            x = []
            y = []
            pdb_list = []
            with open(
                    "../PPLM_dataset/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv") as f:
                csv_reader = csv.reader(f)
                next(csv_reader)
                data = []
                for i, line in enumerate(csv_reader):
                    if i < 100000:
                        try:
                            seq = torch.tensor(dictionary.encode(line[-10]))
                            ddG = torch.tensor(float(line[-3]))
                            pdb = line[-8].split('.')[0].replace("|", ":")
                            x.append(seq)
                            y.append(ddG)
                            pdb_list.append(pdb)
                        except:
                            print(f"error evaluating Line {i} {line[-10]} {line[-3]}")

        full_dataset = Dataset(x,y,pdb_list)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=lambda x: collate_fn(x,discriminator_type))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               collate_fn=lambda x: collate_fn(x,discriminator_type))

    data1 = next(iter(train_loader))
    optimizer = torch.optim.AdamW(discriminator.classifier_head.parameters(), lr=learning_rate)


    test_losses = []
    test_accuracies = []

    # training
    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            device=device
        )

        test_loss, test_accuracy = evaluate_performance(
            data_loader=test_loader,
            discriminator=discriminator,
            device=device
        )

        end = time.time()
        print(f"Epoch took: {end - start:.3f}s")

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if save_model:
            torch.save(discriminator.get_classifier().state_dict(),
                       os.path.join(output_fp, f'classifier_head_epoch_{epoch+1}.pt'))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Train a discriminator on top of ESM-IF"
    # )
    # parser.add_argument()
    # args = parser.parse_args()
    #
    train_discriminator(
        dataset='Stability',

    )