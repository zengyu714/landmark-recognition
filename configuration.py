from easydict import EasyDict

CONF = EasyDict()


# Loading data
# ===================================================
CONF.data_root = "/home/kimmy/dataset/"
CONF.toy_file = CONF.data_root + "train_toy.csv"
CONF.data_file = CONF.data_root + "train.csv"
# 4132914 = 3719623 (train) + 247975 (val) + 165316 (test)
CONF.data_split = {'train': 0.9, 'val': 0.06, 'test': 0.04}

CONF.batch_size = 256
CONF.image_size = (96, 96)

# any positive number to slice the training loader
CONF.num_total = -1  # -1 means training all
CONF.num_workers = 4
# ===================================================


# Training
CONF.print_every = 100
CONF.lr = 1e-3
CONF.tot_epochs = 50