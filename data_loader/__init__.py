import pandas as pd
from torch.utils.data import DataLoader
import random

from data_loader import movielens

def make_data_loader(args, **kwargs):
    if args.dataset == "movielens-20m":
        train_list = []
        test_list = []

        imdb = pd.read_csv(args.data_path + "/imdb.csv")

        genres_lookup = {}
        actors_lookup = {}
        directors_lookup = {}
        languages_lookup = {}

        for index, row in imdb.iterrows():
            genres = row["genres"]
            actors = row["actors"]

        train_set = movielens.MovieLensDataset(args, data_list=train_list, split="train")
        test_set = movielens.MovieLensDataset(args, data_list=test_list, split="test")

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, test_loader

    else:
        raise NotImplementedError