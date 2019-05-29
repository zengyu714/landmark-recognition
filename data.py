"""
- training_dataset: 4132914
- landmark_id: 203094 uniques classes (hist from 1 to 10247 per classes)

- toy_dataset: 249190

"""

import multiprocessing
import os

from PIL import Image
import numpy as np
import pandas as pd
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T

from config import Config
from utils.util import parse_info, id_to_path

CONF = Config(data_root="/home/kimmy/dataset")
DATA_ROOT = CONF.data_root

TRAIN_FILE = CONF.train_file
TRAIN_ROOT = CONF.train_root

TEST_FILE = CONF.test_file
TEST_ROOT = CONF.test_root

TOY_FILE = CONF.toy_file
if CONF.training_toy_dataset:
    TRAIN_FILE = TOY_FILE

DATA_SPLIT = CONF.data_split
MIN_SAMPLES = CONF.min_samples

NUM_TOTAL = CONF.num_total

NUM_WORKERS = multiprocessing.cpu_count()


def load_transform(input_size=(96, 96)):
    transform = {
        'train': T.Compose([
            # width = height
            T.Resize(input_size),
            T.RandomHorizontalFlip(),
            T.RandomChoice([
                T.RandomResizedCrop(input_size[0]),
                T.ColorJitter(0.2, 0.2, 0.2, 0.2),
                T.RandomAffine(degrees=15, translate=(0.2, 0.2),
                               scale=(0.8, 1.2), shear=15,
                               resample=Image.BILINEAR)]),

            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]),
        'val'  : T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]),
        'test' : T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    }
    return transform


def sample_toy_dataset(sample_size=2000, savename=TOY_FILE):
    """
    Args:
        sample_size: # of sampled classes
    """

    df = pd.read_csv(TRAIN_FILE)

    counter = df.landmark_id.value_counts()
    landmark_ids, counts = counter.index, counter.values
    sampled_ids = np.random.choice(landmark_ids, size=sample_size, replace=False, p=counts / sum(counts))
    sampled_df = df[df.landmark_id.isin(sampled_ids)]
    sampled_df.to_csv(savename)


def relabel(df, save_mapping=True):
    """
        Save the relabeled csv.
    """
    df.drop(columns='url', inplace=True)
    counts = df.landmark_id.value_counts()
    selected_classes = counts[counts >= MIN_SAMPLES].index
    num_classes = selected_classes.shape[0]

    landmarks_frame = df.loc[df.landmark_id.isin(selected_classes)].copy()

    mapping_dict = {cls: i for i, cls in enumerate(selected_classes)}
    landmarks_frame.landmark_id = landmarks_frame.landmark_id.map(mapping_dict)

    savename = f"{CONF.data_root}/mapping.npy"
    savename_inv = f"{CONF.data_root}/mapping_inv.npy"

    if save_mapping and not os.path.exists(savename):
        np.save(savename, mapping_dict)
        inv_dict = {v: k for k, v in mapping_dict.items()}
        np.save(savename_inv, inv_dict)

    return num_classes, landmarks_frame


class LandmarkDataset(Dataset):
    """Google landmark dataset
    Ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, csv_file, root_dir, input_size, stage, split=DATA_SPLIT):
        """
        csv_file(str) : path of csv file with url, file names and landmark_id
        root_dir(str) : path of images folder
        stage (str): should be in ['train', 'val', 'test']
        """
        df = pd.read_csv(csv_file)[:NUM_TOTAL]
        self.num_classes, self.landmarks_frame = relabel(df, save_mapping=True)
        num_tot = self.landmarks_frame.shape[0]

        # num_train, num_val, num_test
        sections = np.round(num_tot * np.cumsum([split[k] for k in ['train', 'val', 'test']])).astype(int)
        self.train_idx, self.val_idx, self.test_idx, *_ = np.split(np.arange(num_tot), sections)

        self.root_dir = root_dir
        self.index = eval(f"self.{stage}_idx")

        self.transform = load_transform(input_size)[stage]

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


def load_dataset(input_size, batch_size):
    landmark_train = LandmarkDataset(csv_file=TRAIN_FILE, root_dir=TRAIN_ROOT, stage='train', input_size=input_size)
    # "squeeze" the huge training dataset to make it save checkpoint / print logs timely
    # approximates 15
    chunk_nums = DATA_SPLIT['train'] // DATA_SPLIT['val']
    landmark_train_subsets = random_split(landmark_train, chunk_nums)
    loader_train_sets = [DataLoader(landmark_train_subset, batch_size=batch_size, num_workers=NUM_WORKERS,
                                    collate_fn=lm_collate, pin_memory=True)
                         for landmark_train_subset in landmark_train_subsets]

    landmark_val = LandmarkDataset(csv_file=TRAIN_FILE, root_dir=TRAIN_ROOT, stage='val', input_size=input_size)
    loader_val = DataLoader(landmark_val, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                            collate_fn=lm_collate, pin_memory=True)

    landmark_test = LandmarkDataset(csv_file=TRAIN_FILE, root_dir=TRAIN_ROOT, stage='test', input_size=input_size)
    loader_test = DataLoader(landmark_test, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                             collate_fn=lm_collate, pin_memory=True)

    num_classes = landmark_train.num_classes
    return loader_train_sets, loader_val, loader_test, num_classes


def lm_collate_submit(batch):
    image = torch.stack([item['image'] for item in batch if item is not None])
    name = [item['name'] for item in batch if item is not None]

    return {'image': image, 'name': name}


class LandmarkDatasetSubmit(Dataset):
    """Google landmark dataset
    Ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, csv_file, root_dir, input_size):
        """
        csv_file(str) : path of csv file with url, file names and landmark_id
        root_dir(str) : path of images folder
        """
        df = pd.read_csv(csv_file)
        try:
            df.drop(columns='url', inplace=True)
        except KeyError:
            pass

        self.num_tot = df.shape[0]
        self.root_dir = root_dir

        self.landmarks_frame = df
        self.transform = load_transform(input_size)['val']

    def __len__(self):
        return self.num_tot

    def __getitem__(self, idx):
        img_name = str(self.landmarks_frame.iloc[idx]['id'])
        img_path = id_to_path(self.root_dir, img_name)

        try:
            image = io.imread(img_path)
        except (FileNotFoundError, OSError, IndexError, UserWarning):
            # print(img_name)
            return None

        if self.transform:
            try:
                image = self.transform(T.ToPILImage()(image))
            except (RuntimeError, ValueError):
                # print(image.shape)
                return None

        return {'image': image, 'name': img_name}


def load_dataset_submit(root_dir, csv_file, input_size, batch_size, num_workers):
    landmark_train_submit = LandmarkDatasetSubmit(csv_file=csv_file, root_dir=root_dir, input_size=input_size)
    loader_submit = DataLoader(landmark_train_submit, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                               collate_fn=lm_collate_submit,
                               pin_memory=True)
    return loader_submit


if __name__ == "__main__":
    # sample_toy_dataset()
    relabel(pd.read_csv(TRAIN_FILE), save_mapping=True)

    # print("Test training loader...")
    loader_train_sets, loader_val, loader_test, _ = load_dataset((96, 96), 16)
    sample = next(iter(loader_val))
    print(sample['image'].shape, sample['landmark_id'])
    #
    # print("Test submitting loader...")
    # loader_submit = load_dataset_submit((96, 96), 256)
    # sample = next(iter(loader_submit))
    # print(sample['image'].shape)
