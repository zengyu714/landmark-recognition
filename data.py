"""
- training_dataset: 4132914
- landmark_id: 203094 uniques classes (hist from 1 to 10247 per classes)

- toy_dataset: 249190

"""

import os
import torch
import pandas as pd
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
from utils.util import parse_info
from configuration import CONF

DATA_ROOT = CONF.data_root
TOY_FILE = CONF.toy_file
DATA_FILE = CONF.data_file
if CONF.training_toy_dataset:
    DATA_FILE = TOY_FILE

DATA_SPLIT = CONF.data_split
IMAGE_SIZE = CONF.image_size
NUM_TOTAL = CONF.num_total

NUM_WORKERS = CONF.num_workers

transform_train = T.Compose([
    # width = height
    T.RandomResizedCrop(IMAGE_SIZE[0]),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Lambda(lambda x: x.div(255.0))
])

transform_val = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Lambda(lambda x: x.div(255.0))
])


def sample_toy_dataset(sample_size=2000, savename=TOY_FILE):
    """
    Args:
        sample_size: # of sampled classes
    """

    train_csv_path = "/home/dataset/train.csv"
    df = pd.read_csv(train_csv_path)

    counter = df.landmark_id.value_counts()
    landmark_ids, counts = counter.index, counter.values
    sampled_ids = np.random.choice(landmark_ids, size=sample_size, replace=False, p=counts / sum(counts))
    sampled_df = df[df.landmark_id.isin(sampled_ids)]
    sampled_df.to_csv(savename)


class LandmarkDataset(Dataset):
    """Google landmark dataset
    Ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, csv_file, root_dir, stage, split=DATA_SPLIT):
        """
        csv_file(str) : path of csv file with url, file names and landmark_id
        root_dir(str) : path of images folder
        stage (str): should be in ['train', 'val', 'test']
        """
        self.landmarks_frame = pd.read_csv(csv_file)[:NUM_TOTAL]
        num_tot = len(self.landmarks_frame)
        # num_train, num_val, num_test
        sections = np.round(num_tot * np.cumsum([split[k] for k in ['train', 'val', 'test']])).astype(int)
        self.train_idx, self.val_idx, self.test_idx, *_ = np.split(np.arange(num_tot), sections)

        self.root_dir = root_dir
        self.index = eval(f"self.{stage}_idx")

        self.transform = transform_train
        if stage in ['val', 'test']:
            self.transform = transform_val

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # offset by current index from different stages
        new_idx = idx + self.index[0]
        img_name, landmark_id = parse_info(self.landmarks_frame, new_idx, self.root_dir)

        # if not os.path.exists(img_name):
        #     return None

        while not os.path.exists(img_name):
            print(f"{img_name} is not found...")
            idx += 1
            new_idx = idx % len(self.index) + self.index[0]
            img_name, landmark_id = parse_info(self.landmarks_frame, new_idx, self.root_dir)

        image = io.imread(img_name)
        if self.transform:
            image = self.transform(T.ToPILImage()(image))

        return {'image': image, 'landmark_id': landmark_id}


def lm_collate(batch):
    image = torch.stack([item['image'] for item in batch if item is not None])
    label = torch.squeeze(torch.stack([torch.LongTensor([item['landmark_id']])
                                       for item in batch if item is not None]), dim=1)
    return {'image': image, 'landmark_id': label}


def random_split(dataset, chunk_nums):
    """
    Randomly split a dataset into non-overlapping new datasets of given chunk nums.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    indices = torch.randperm(len(dataset)).tolist()
    size = len(dataset) // chunk_nums
    chunks = []

    chunk_nums = int(chunk_nums)
    size = int(size)
    for i in range(chunk_nums - 1):
        chunks.append(Subset(dataset, indices[i * size: (i + 1) * size]))
    # append the last chunk
    return chunks + [Subset(dataset, indices[(chunk_nums - 1) * size:])]


def load_dataset(batch_size=128):
    landmark_train = LandmarkDataset(csv_file=DATA_FILE, root_dir=DATA_ROOT, stage='train')
    # "squeeze" the huge training dataset to make it save checkpoint / print logs timely
    # approximates 15
    chunk_nums = DATA_SPLIT['train'] // DATA_SPLIT['val']
    landmark_train_subsets = random_split(landmark_train, chunk_nums)
    loader_train_sets = [DataLoader(landmark_train_subset, batch_size=batch_size, num_workers=NUM_WORKERS,
                                    collate_fn=lm_collate, pin_memory=True)
                         for landmark_train_subset in landmark_train_subsets]

    landmark_val = LandmarkDataset(csv_file=DATA_FILE, root_dir=DATA_ROOT, stage='val')
    loader_val = DataLoader(landmark_val, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                            collate_fn=lm_collate, pin_memory=True)

    landmark_test = LandmarkDataset(csv_file=DATA_FILE, root_dir=DATA_ROOT, stage='test')
    loader_test = DataLoader(landmark_test, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                             collate_fn=lm_collate, pin_memory=True)

    return loader_train_sets, loader_val, loader_test


if __name__ == "__main__":
    # sample_toy_dataset()
    loader_train_sets, loader_val, loader_test = load_dataset()
    sample = next(iter(loader_val))
    print(sample['image'].shape, sample['landmark_id'])
