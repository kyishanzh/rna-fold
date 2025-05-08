import logging
import pickle
import os
import sys

import numpy as np
import torch

from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils import get_device, save_ss2ct, timing
from rhofold.relax.relax import AmberRelaxation
from rhofold.utils.alphabet import get_features


from distributed_utils import (
    get_world_size,
    get_rank,
    is_main_process,
    init_distributed_mode,
)
import torch.distributed as dist
import pandas as pd


def load_model(ckpt="./pretrained/RhoFold_pretrained.pt"):
    model = RhoFold(rhofold_config)
    model.load_state_dict(
        torch.load(ckpt, map_location=torch.device("cpu"))["model"], strict=False
    )
    model.eval()

    return model.cuda()


data_folder = "/dev/shm/"

train_csv = data_folder + "train_sequences.csv"
val_csv = data_folder + "validation_sequences.csv"
test_csv = data_folder + "test_sequences.csv"


def prepare_ds():
    """
    Prepare the dataset for training, validation, and testing.
    Returns:
        id_to_seq: dict, key is the sequence id, value is the sequence
        id_to_loc: dict, key is the sequence id,
            value is (N, 3) tensor, N is the number of atoms, 3 is the x, y, z coordinates
    """

    id_to_seq = dict()

    def prepare_csv(csv_path):
        df = pd.read_csv(csv_path)
        for idx, row in df.iterrows():
            seq_id = row["target_id"]
            # print(seq_id)
            seq = row["sequence"]
            # only need to do once.
            with open(data_folder + f"MSA/{seq_id}.fasta", "w") as f:
                f.write(f">{seq_id}\n{seq}")
            id_to_seq[seq_id] = seq

    prepare_csv(train_csv)
    prepare_csv(val_csv)
    prepare_csv(test_csv)

    train_labels = data_folder + "train_labels.csv"
    val_labels = data_folder + "validation_labels.csv"
    id_to_loc = dict()
    for csv in [train_labels, val_labels]:
        df = pd.read_csv(csv)
        for idx, row in df.iterrows():
            id = row["ID"]
            id = "_".join(id.split("_")[:-1])
            x_1, y_1, z_1 = row["x_1"], row["y_1"], row["z_1"]
            if id not in id_to_loc:
                id_to_loc[id] = [(x_1, y_1, z_1)]
            else:
                id_to_loc[id].append((x_1, y_1, z_1))

    for idx in id_to_loc.keys():
        id_to_loc[idx] = torch.tensor(id_to_loc[idx], dtype=torch.float32)
    return id_to_seq, id_to_loc


id_to_seq, id_to_loc = prepare_ds()


def split_to_csv(csv_path, split="train"):  # train, val, test
    if split == "train":
        csv_path = train_csv
    elif split == "val":
        csv_path = val_csv
    elif split == "test":
        csv_path = test_csv
    return csv_path


def all_seq_ids(split="train"):  # train, val, test; return all the seq ids
    csv_path = split_to_csv(split)
    df = pd.read_csv(csv_path)
    idx_list = df["target_id"].tolist()
    return [idx for idx in idx_list if len(id_to_seq[idx]) < 45]


def g_features(seq_id):
    fasta_path = f"{data_folder}MSA/{seq_id}.fasta"  # This should contain just your query sequence
    msa_path = f"{data_folder}MSA/{seq_id}.MSA.fasta"  # This is your MSA file

    # Get features (optionally specify msa_depth, default is 128)
    features = get_features(
        fas_fpath=fasta_path,
        msa_fpath=msa_path,
        msa_depth=128,  # You can adjust this if you want to use more/fewer sequences from the MSA
    )
    evo2_fea = evo2_features[seq_id]

    return {
        "seq": features["seq"],
        "tokens": features["tokens"].cuda(),
        "rna_fm_tokens": features["rna_fm_tokens"].cuda(),
        "evo2_fea": evo2_fea.cuda(),
    }


def kabsch_align(P, Q):
    # P, Q: [L, 3]
    P_mean = P.mean(dim=0)
    Q_mean = Q.mean(dim=0)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean

    H = P_centered.T @ Q_centered
    U, S, Vt = torch.svd(H)
    R = Vt @ U.T
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt @ U.T
    P_aligned = P_centered @ R
    return P_aligned, Q_centered


def tm_score(P, Q):
    # P: predicted coords, Q: true coords, both [L, 3]
    L = P.shape[0]
    d0 = 0.0
    if L < 12:
        d0 = 0.3
    elif L < 16:
        d0 = 0.4
    elif L < 20:
        d0 = 0.5
    elif L < 24:
        d0 = 0.6
    elif L < 30:
        d0 = 0.7
    else:
        d0 = 0.6 * (L - 0.5) ** 0.5 - 2.5

    P_aligned, Q_centered = kabsch_align(P, Q)
    dist = torch.norm(P_aligned - Q_centered, dim=1)
    score = (1 / (1 + (dist / d0) ** 2)).mean()
    return score.item()


import torch.distributed as dist
from tqdm import tqdm

with open("/dev/shm/nvidia_evo2_full_embeddings.pkl", "rb") as f:
    evo2_features = pickle.load(f)


def eval_model(generator):
    """
    generator: a function that takes in features, and obtain a sample
    """
    val_idx = all_seq_ids("val")
    if is_main_process():
        print(f"Evaluating {len(val_idx)} sequences")
    score_list = []
    for i, idx in enumerate(tqdm(val_idx)):
        if i % get_world_size() != get_rank():
            continue
        features = g_features(idx)
        sample = generator(features)
        max_score = 0.0
        for v in sample:
            score = tm_score(v, id_to_loc[idx].to(v.device))
            max_score = max(max_score, score)
        score_list.append(max_score)

    local_sum = torch.tensor(sum(score_list), dtype=torch.float32, device="cuda")
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)

    if is_main_process():
        score = local_sum.item() / len(val_idx)
        print(f"Average score: {score}")
        return score
    else:
        return 0.0


# torchrun --standalone --nproc_per_node=8 utils.py
if __name__ == "__main__":

    init_distributed_mode()
    model = load_model()

    def generator(features):
        outputs = model(**features)
        preds = []
        for i in range(5):
            preds.append(outputs[i]["cords_c1'"][0][0])
        return preds

    eval_model(generator)
