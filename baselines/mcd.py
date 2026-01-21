# -*- coding: utf-8 -*-
import os
import json
from PIL import Image

import torch
import open_clip

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import webdataset as wds

from tqdm import tqdm
import argparse, glob, os, random


def parse_args():
    p = argparse.ArgumentParser("Monte‑Carlo‑Dropout attack for CLIP")
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--pretrained", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=32)

    # shards
    p.add_argument("--eval_member_shards", required=True, nargs='+')
    p.add_argument("--eval_nonmember_shards", required=True, nargs='+')

    # MC‑Dropout params
    p.add_argument("--mc_passes", type=int, default=10,
                   help="# stochastic forward passes per sample")
    p.add_argument("--dropout_p", type=float, default=0.05,
                   help="drop probability to set in attention Dropout layers")

    # threshold selection
    p.add_argument("--fpr_target", type=float, default=0.01)

    return p.parse_args()


args = parse_args()


def eval_mia(sim_member, sim_non_member):
    # Create the prediction and ground truth arrays
    # Members are labeled as 1 and non-members as 0.
    predictions = np.concatenate([sim_member, sim_non_member])
    ground_truth = np.concatenate([np.ones(len(sim_member)), np.zeros(len(sim_non_member))])

    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions)

    # Calculate the Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Extract the True Positive Rate (TPR) at a False Positive Rate (FPR) below 1%
    indices = np.where(fpr < 0.01)[0]
    if len(indices) > 0:
        tpr_at_1_fpr = tpr[indices[-1]]  # Use the last index where FPR is below 1%
    else:
        tpr_at_1_fpr = 0.0  # If no index meets the criteria, default to 0

    return roc_auc,tpr_at_1_fpr


"""## Download datasets

We use LAION400M as member samples and CC3M as non-member samples. The data is downloaded from HuggingFace datasets (CC3M) and an anonymous bucket (LAION400M). This is only a small set of samples for demonstration.
"""


"""## Define WebDataloader class"""

class WebDataset(Dataset):
    def __init__(self, tar_file, num_samples=None, transform=None):

        self.transform = transform
        self.tar_file = tar_file

        # Create a WebDataset that decodes images (as PIL) and extracts text.
        dataset = wds.WebDataset(tar_file, shardshuffle=False).decode("pil").to_tuple("jpg", "txt")

        self.samples = list(dataset)
        if num_samples:
            self.samples = self.samples[:num_samples]

        print(f"Loaded {len(self.samples)} samples from {tar_file}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, caption = self.samples[idx]
        if self.transform:
            image = self.transform(image)
        # Wrap the caption in a list to match the COCO interface.
        return image, [caption]

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to return a dictionary with images and captions."""
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        return {"images": images, "captions": captions}


class WebDataLoader:
    def __init__(self, tar_file, batch_size=32, num_samples=None, transform=None):
        """
        A DataLoader wrapper with a similar interface to the COCODataLoader.

        Args:
            tar_file (str): Path to the tar file containing your webdataset.
            batch_size (int): Batch size for the DataLoader.
            num_samples (int, optional): If provided, limits the dataset to this many samples.
            transform (callable, optional): Transformations to apply to each image.
        """
        self.tar_file = tar_file
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.transform = transform

        self.dataset = None
        self.dataloader = None

    def load_data(self):
        self.dataset = WebDataset(
            tar_file=self.tar_file,
            num_samples=self.num_samples,
            transform=self.transform
        )
        print(f"Loaded dataset with {len(self.dataset)} samples.")

    def get_dataloader(self):
        if self.dataset is None:
            self.load_data()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=WebDataset.collate_fn
        )
        return self.dataloader

"""## Create Dataloaders"""

member_dataloader = WebDataLoader(tar_file=args.eval_member_shards, batch_size=args.batch_size).get_dataloader()
non_member_dataloader = WebDataLoader(tar_file=args.eval_nonmember_shards, batch_size=args.batch_size).get_dataloader()

"""## Baseline results

We create the deterministic CLIP model and compute cosine similarities for member and non_member datasets.
"""

device = args.device

model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=args.model,
            pretrained=args.pretrained
        )

tokenizer = open_clip.get_tokenizer(args.model)
_ = model.to(device)
_ = model.eval() # Deterministic CLIP

deterministic_scores = {'member': [], 'non_member': []}

for key, dataloader in [('member', member_dataloader), ('non_member', non_member_dataloader)]:
    for batch in tqdm(dataloader, desc=f"{key} batches"):
        batch_images = torch.stack([preprocess(item.convert("RGB")) for item in batch['images']])
        batch_images = batch_images.to(device)

        batch_texts_tok = tokenizer([text for texts in batch['captions'] for text in texts]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(batch_images)
            text_features = model.encode_text(batch_texts_tok)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            scores = image_features @ text_features.T
            deterministic_scores[key].append(scores.diagonal().cpu().detach().numpy())

"""## MCDropout-MIA

Perform the MCDropout-MIA attack: Create the CLIP model, set dropout rate to 0.05 on all attention layers, and run 10 forward passes along the entire dataset. Keep the pairwise similarities for each pass.
"""

model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=args.model,
            pretrained=args.pretrained
        )
tokenizer = open_clip.get_tokenizer(args.model)

# Enable dropout in the attention layers
for resblock in model.visual.transformer.resblocks:
  resblock.attn.dropout = args.dropout_p
for resblock in model.transformer.resblocks:
  resblock.attn.dropout = args.dropout_p

_ = model.to(device)

NUM_DRAWS = args.mc_passes  # number of stochastic forward passes
mcd_scores = {'member': [], 'non_member': []}

for draw in tqdm(range(NUM_DRAWS), desc="MCDropout Draws", position=0, leave=True):
    for key, dataloader in [('member', member_dataloader), ('non_member', non_member_dataloader)]:
        batch_scores = []
        desc = f"{key.capitalize()} Batches"
        for batch in tqdm(dataloader, desc=desc, position=1, leave=False):
            batch_images = torch.stack([preprocess(item.convert("RGB")) for item in batch['images']])
            batch_images = batch_images.to(device)

            batch_texts_tok = tokenizer([text for i, texts in enumerate(batch['captions']) for text in texts]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(batch_images)
                text_features = model.encode_text(batch_texts_tok)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                scores = image_features @ text_features.T
                batch_scores.append(scores.diagonal().cpu().detach().numpy())

        mcd_scores[key].append(batch_scores)

"""## Aggregate similarities from MCDropout draws

"""

deterministic_scores['member'] = np.concatenate(deterministic_scores['member'])
deterministic_scores['non_member'] = np.concatenate(deterministic_scores['non_member'])

print(deterministic_scores['member'].shape, deterministic_scores['non_member'].shape)


# take the mean similarity across MCD passes for each image-text pair

for i,scores in enumerate(mcd_scores['member']):
  mcd_scores['member'][i] = np.concatenate(scores)
mcd_scores['member'] = np.mean(mcd_scores['member'], axis=0)

for i,scores in enumerate(mcd_scores['non_member']):
  mcd_scores['non_member'][i] = np.concatenate(scores)
mcd_scores['non_member'] = np.mean(mcd_scores['non_member'], axis=0)

print(mcd_scores['member'].shape, mcd_scores['non_member'].shape)

"""## Plot cosine similarity distributions"""

fig, axs = plt.subplots(1, 2, figsize=(10, 2), sharey=True)

for ax, scores in zip(axs,[deterministic_scores, mcd_scores]):

    range_score = (min(scores['member'].min(), scores['non_member'].min()),
                   max(scores['member'].max(), scores['non_member'].max()))

    # Plotting the distributions
    ax.hist(scores['member'], bins=30, range=range_score, alpha=0.5, label="Member", color='blue')
    ax.hist(scores['non_member'], bins=30, range=range_score, alpha=0.7, label="Non-member", color='pink')

    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Samples')
    ax.legend()
    ax.grid(True)

axs[0].set_title('Deterministic CLIP')
axs[1].set_title('MCDropout (avg)')

"""## Compute AUC and TPR@1%FPR"""

def eval_mia(sim_member, sim_non_member):
    # Create the prediction and ground truth arrays
    # Members are labeled as 1 and non-members as 0.
    predictions = np.concatenate([sim_member, sim_non_member])
    ground_truth = np.concatenate([np.ones(len(sim_member)), np.zeros(len(sim_non_member))])

    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions)

    # Calculate the Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Extract the True Positive Rate (TPR) at a False Positive Rate (FPR) below 1%
    indices = np.where(fpr < args.fpr_target)[0]
    if len(indices) > 0:
        tpr_at_1_fpr = tpr[indices[-1]]  # Use the last index where FPR is below 1%
    else:
        tpr_at_1_fpr = 0.0  # If no index meets the criteria, default to 0

    return roc_auc,tpr_at_1_fpr

print('CSA Baseline')
roc_auc,tpr_at_1_fpr = eval_mia(deterministic_scores['member'], deterministic_scores['non_member'])
print(f"  AUC: {roc_auc:.4f}")
print(f"  TPR@1%FPR: {tpr_at_1_fpr:.4f}")

print('\nMCDropout-MIA (avg)')
roc_auc,tpr_at_1_fpr = eval_mia(mcd_scores['member'], mcd_scores['non_member'])
print(f"  AUC: {roc_auc:.4f}")
print(f"  TPR@1%FPR: {tpr_at_1_fpr:.4f}")

