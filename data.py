"""
- training_dataset: 4132914
- landmark_id: 203094 uniques classes (hist from 1 to 10247 per classes)

- toy_dataset: 249190

"""

import os
import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms as T
from util import parse_info

DATA_ROOT = "/home/kimmy/dataset/"
TOY_FILE = DATA_ROOT + "train_toy.csv"

BATCH_SIZE = 64
NUM_TOTAL = 20000

NUM_TRAIN = int(NUM_TOTAL * 0.9)
NUM_VAL = int(NUM_TOTAL * 0.05)
NUM_TEST = NUM_TOTAL - NUM_TRAIN - NUM_VAL

NUM_WORKERS = 4

transform_train = T.Compose([
    T.RandomResizedCrop(64),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225])
])
transform_val = T.Compose([
    T.Resize(64),
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225])
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

    def __init__(self, csv_file, root_dir, stage, split={'train': 0.9, 'val': 0.06, 'test': 0.04}):
        """
        csv_file(str) : path of csv file with url, file names and landmark_id
        root_dir(str) : path of images folder
        stage (str): should be in ['train', 'val', 'test']
        """
        self.landmarks_frame = pd.read_csv(csv_file)
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

        while not os.path.exists(img_name):
            print(f"{img_name} is not found...")
            new_idx = (idx + 1) % len(self.index) + self.index[0]
            img_name, landmark_id = parse_info(self.landmarks_frame, new_idx, self.root_dir)

        image = io.imread(img_name)
        if self.transform:
            image = self.transform(T.ToPILImage()(image))

        return {'image': image, 'landmark_id': landmark_id}


def load_dataset():
    landmark_train = LandmarkDataset(csv_file=TOY_FILE, root_dir=DATA_ROOT, stage='train')
    loader_train = DataLoader(landmark_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    landmark_val = LandmarkDataset(csv_file=TOY_FILE, root_dir=DATA_ROOT, stage='val')
    loader_val = DataLoader(landmark_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    landmark_test = LandmarkDataset(csv_file=TOY_FILE, root_dir=DATA_ROOT, stage='test')
    loader_test = DataLoader(landmark_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return loader_train, loader_val, loader_test


if __name__ == "__main__":
    # sample_toy_dataset()
    loader_train, loader_val, loader_test = load_dataset()
    sample = next(iter(loader_train))
    print(sample['image'].shape, sample['landmark_id'])
