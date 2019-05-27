"""
Data configuration
"""

import os


class Config:
    def __init__(self, data_root="/home/kimmy/dataset"):
        # Project
        # ===================================================
        self.proj_root = os.getcwd()
        self.ckpt_root = os.path.join(self.proj_root, "checkpoints")
        self.submit_root = os.path.join(self.proj_root, "submission")
        # ===================================================

        # Data
        # ===================================================
        self.data_root = data_root
        self.toy_file = os.path.join(self.data_root, "train_toy.csv")
        self.training_toy_dataset = False

        self.train_file = os.path.join(self.data_root, "train.csv")
        self.test_file = os.path.join(self.data_root, "test.csv")

        self.train_root = os.path.join(self.data_root, "train")
        self.test_root = os.path.join(self.data_root, "test")

        # 4132914 = 37123 (train) + 247975 (val) + 165316 (test)
        self.data_split = {'train': 0.9, 'val': 0.06, 'test': 0.04}
        self.min_samples = 50

        # any positive number to slice the training loader
        self.num_total = -1  # -1 means training all
        self.print_every = 100
        # ===================================================

        # Delf
        # ===================================================
        self.train_delf = self.data_root + "train_delf/"
        self.test_delf_root = self.data_root + "test_delf/"
        self.cls_root = self.data_root + "cls/"
        # ===================================================
