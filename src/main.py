import yaml
import os
import random
import time
import warnings
import sys
import shutil
import argparse
import pickle
import warnings
import time
import csv
import pandas as pd

warnings.filterwarnings("ignore")

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch
import torch.optim as optim

from data.data_manager import DataWrapper, DataManager

from modeling.wscnnlstm import WSCNNLSTM
from engine.train import Train
from utils import *


class Runner:
    def __init__(self, args):
        config_file = args.config
        with open(config_file) as yml:
            config = yaml.load(yml, Loader=yaml.FullLoader)

        #self.train_ratio = float(args.ratio)
        self.set_num = int(args.set)

        self.out_dir = f"{config['DATA']['out_dir']}/set{self.set_num}/"
        os.makedirs(self.out_dir, exist_ok=True)
        self.data_config = config["DATA"]["data_config"]

        self.gamma = float(config["MODEL"]["gamma"])
        self.eta = float(config["MODEL"]["eta"])
        self.opt_lr = float(config["MODEL"]["optimizer_lr"])
        self.wd = float(config["MODEL"]["weight_decay"])
        self.earlystop = int(config["MODEL"]["earlystop"])

        self.batch_size = int(config["MODEL"]["batch"])
        self.EPOCH = int(config["MODEL"]["epoch"])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def save_dict(self, data, filename):
        outfile = self.out_dir + filename + ".csv"
        df = list()
        for idx in range(len(data["X"])):
            df.append(["".join(data["X"][idx]), data["Y"][idx]])
        df = pd.DataFrame(df, columns=["X", "Y"])
        df.to_csv(outfile, sep="\t", index=False, header=False)
        print(f"Saved {filename} data ...")

    def dataload(self, args):
        # encoding -> rand_augment
        DM = DataManager(self.batch_size, self.data_config, args)
        ret1, ret2, ret3 = DM.target_load()
        #ret1, ret2, ret3 = DM.merge_load()

        self.save_dict(ret1, "train")
        self.save_dict(ret2, "valid")
        self.save_dict(ret3, "test")

        #self.source_loader = DM.loader_only(res)
        self.train_loader = DM.loader_only(ret1)
        self.val_loader = DM.loader_only(ret2)
        self.test_loader = DM.loader_only(ret3)

        return DM.seqlen

    def init_model(self, len):
        # Create model
        self.framework = WSCNNLSTM(len).to(self.device)

        self.optimizers = optim.SGD(
            self.framework.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4
        )

        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizers,
            first_cycle_steps=5,
            cycle_mult=1.0,
            max_lr=5e-2,
            min_lr=1e-4,
            warmup_steps=2,
            gamma=1.0,
        )

    def train_model(self, logger):

        Engine = Train(self)
        Engine.run_step(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="configuration file path")
    parser.add_argument(
        "--target", type=str, help="target domain for Domain adaptation"
    )
    # parser.add_argument(
    #     "--source", type=str, help="source domain for Domain adaptation"
    # )
    parser.add_argument("--set", type=int, help=">1")
    #parser.add_argument("--ratio", type=float, help="train ratio", default=1.0)

    args = parser.parse_args()

    start = time.time()
    runner = Runner(args)
    logger = CompleteLogger(runner.out_dir)

    seqlen = runner.dataload(args)
    runner.init_model(seqlen)

    runner.train_model(logger)
    end = time.time()
    print("time elapsed:", end - start)

    logger.close()
