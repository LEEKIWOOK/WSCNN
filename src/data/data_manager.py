import csv
import math
import yaml
import random
import pandas as pd
import numpy as np
from Bio.Seq import Seq
from functools import reduce
import pickle
import PIL

import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from utils import *


class DataManager:
    def __init__(self, batch_size, data_config, args):
        with open(data_config) as yml:
            config = yaml.load(yml, Loader=yaml.FullLoader)

        data_cfg = config["DATA"]
        self.domain_list = [
            "cas9_wt_kim",
            "cas9_wt_wang",
            "cas9_wt_xiang",
            "cas9_hf_wang",
            "cas9_esp_wang",
            "cas9_hct116_hart",  # 0.3
            "cas9_hl60_wang",  # 0.87
            "cas9_hek293t_doench",  # -0.01
            "cas9_hela_hart",  # 0.35
        ]
        self.domain_file = [
            f"{data_cfg['in_dir']}/{data_cfg[x]}" for x in self.domain_list
        ]
        # self.target_domain = self.domain_list[args.target]
        # self.target_file = self.domain_file[args.target]
        self.target_list = args.target.split(",")
        self.target_domain = list()
        self.target_file = list()

        self.random_seed = data_cfg["seed"]
        self.batch_size = batch_size
        self.seqlen = 33

    def target_load(self, ratio=1.0):

        
        data = pickle.load(open(self.domain_file[int(self.target_list[0])], "rb"))
        
        

        data_size = len(data["X"])
        indice = list(range(data_size))

        np.random.shuffle(indice)

        minY = min(data["Y"])
        maxY = max(data["Y"])
        data["Y"] = [(i - minY) / (maxY - minY) for i in data["Y"]]

        test_ratio = 0.15
        val_ratio = test_ratio

        test_size = int(np.floor(data_size * test_ratio))
        tv_size = int(np.floor(data_size * (1 - test_ratio) * ratio))

        train_size = int(np.floor(tv_size * (1 - val_ratio)))
        valid_size = int(np.floor(tv_size * val_ratio))

        indices = dict()
        indices["Val"] = random.sample(indice[:valid_size], valid_size)
        indices["Test"] = random.sample(
            indice[valid_size : valid_size + test_size], test_size
        )
        indices["Train"] = random.sample(
            indice[valid_size + test_size : valid_size + test_size + train_size],
            train_size,
        )

        train_set = {
            "X": [data["X"][i] for i in indices["Train"]],
            "Y": [data["Y"][i] for i in indices["Train"]],
        }
        test_set = {
            "X": [data["X"][i] for i in indices["Test"]],
            "Y": [data["Y"][i] for i in indices["Test"]],
        }
        valid_set = {
            "X": [data["X"][i] for i in indices["Val"]],
            "Y": [data["Y"][i] for i in indices["Val"]],
        }
        return train_set, valid_set, test_set

    def loader_only(self, data):

        loader = DataLoader(
            DataWrapper(data),
            batch_size=self.batch_size,
            num_workers=8,
            drop_last=True,
        )
        print(f"Size : {len(loader) * self.batch_size}")
        return loader


class DataWrapper:
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.kmer = 3
        self.kmer_stride = 1
        nucleotide = ['A', 'C', 'G', 'T']
        table_key = [x+y+z for x in nucleotide for y in nucleotide for z in nucleotide]
        self.table = {key:idx for idx, key in enumerate(table_key)}

    def __len__(self):
        return len(self.data["X"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        res = dict()
        rlist = list()
        for col in self.data.keys():
            if col == "X":
                for i in range(len(self.data[col][idx]) - (self.kmer - 1)):
                  char = ''.join(self.data[col][idx][i:i + self.kmer])
                  rlist.append(self.table.get(char, -1))
                res[col] = torch.tensor(rlist, dtype=torch.long)
            else:
                res[col] = torch.tensor(self.data[col][idx], dtype=torch.float)
        return res
