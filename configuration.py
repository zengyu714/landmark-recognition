"""
Data configuration
"""

from easydict import EasyDict

CONF = EasyDict()

# Data
# ===================================================
CONF.data_root = "../../"
CONF.data_file = CONF.data_root + "train.csv"
CONF.toy_file = CONF.data_root + "train_toy.csv"
CONF.training_toy_dataset = False

# 4132914 = 37123 (train) + 247975 (val) + 165316 (test)
CONF.data_split = {'train': 0.9, 'val': 0.06, 'test': 0.04}
CONF.image_size = (224, 224)
CONF.min_samples = 183
CONF.max_samples = 10000
# filtered with 18k classes

# any positive number to slice the training loader
CONF.num_total = -1  # -1 means training all

CONF.print_every = 100
# ===================================================


# Testing
# ===================================================
CONF.delf_root = CONF.data_root + "delf/"
# ===================================================
