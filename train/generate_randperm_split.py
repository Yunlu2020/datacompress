import os

import json

import argparse
import torch
from datasets import load_dataset


def main(args):
    data_files = {"train": [f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(14)]}
    
    c4_train = load_dataset("allenai/c4", data_files=data_files, split="train", cache_dir='./dataset/')
    
    dataset_generator = torch.Generator()
    dataset_generator.manual_seed(42)
    randperm = torch.randperm(len(c4_train), generator=dataset_generator)
    
    train_len = int(len(c4_train) * args.ratio)

    train_split_ids = randperm[:train_len].numpy().tolist()
    val_split_ids = randperm[train_len:].numpy().tolist()

    # dataset_generator = torch.Generator()
    # dataset_generator.manual_seed(42)
    # randperm = torch.randperm(len(c4_train), generator=dataset_generator)
    # train_len = int(len(c4_train) * args.ratio)
    # train_split_ids2 = randperm[:train_len].numpy().tolist()
    # val_split_ids2 = randperm[train_len:].numpy().tolist()
    # for i in range(10):
    #     assert val_split_ids2[i] == val_split_ids[i]
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    json.dump(train_split_ids, open(os.path.join(args.save_path, 'train_split_ids.json'), 'w'))
    json.dump(val_split_ids, open(os.path.join(args.save_path, 'val_split_ids.json'), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str, default='./dataset_split/round1/')
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--ratio", type=float, default=0.7)
    args = parser.parse_args()
    
    main(args)