import argparse
import os
import numpy as np
from tqdm import tqdm
import torch

from data_loader import make_data_loader

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Make DataLoader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.test_loader = make_data_loader(args, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Improving Complete the Look Training")

    # DataLoader
    parser.add_argument("--data-path", type=str, default="/SSD/MovieLens/ml-20m", help="path of the dataset")
    parser.add_argument("--workers", type=int, default=4, help="dataloader # of threads")
    parser.add_argument("--dataset", type=str, default="movielens-20m", choices=["movielens-20m"], help="name of the dataset")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size during training")

    args = parser.parse_args()


    trainer = Trainer(args)

if __name__ == "__main__":
    main()