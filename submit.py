import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import load_dataset_submit
from models.resnet import resnet50
from models.seatt import seatt_resnext50_base
from utils.delf.matching import DeLF
from config import Config

CONF = Config(data_root="/home/kimmy/dataset-96")

parser = argparse.ArgumentParser(description='Google Landmark Recognition Challenge')
parser.add_argument('-g', '--cuda-device', type=int, default=0,
                    help='Choose which gpu to use (default: 0)')

args = parser.parse_args()
if torch.cuda.is_available():
    torch.cuda.set_device(args.cuda_device)
    device = torch.cuda.current_device()
else:
    device = torch.device('cpu')

curr_model = seatt_resnext50_base

CHECKPOINT_PATH = os.path.join(CONF.ckpt_root, "seatt_111_best.ckpt")
INV_MAPPING_PATH = os.path.join(CONF.data_root, "mapping_inv.npy")
SAMPLE_PATH = os.path.join(CONF.data_root, "recognition_sample_submission.csv")

TRAIN_DELF = CONF.train_delf
TEST_DELF = CONF.test_delf
CLS_ROOT = CONF.cls_root

OUT_PATH = os.path.join(CONF.submit_root, "submission.csv")
OUT_DELF_PATH = f"{OUT_PATH[:-4]}_delf.csv"
THRESHOLD = 0.4

FILTERED_ROOT = os.path.join(CONF.data_root, "filtered")
if not os.path.exists(FILTERED_ROOT):
    os.makedirs(FILTERED_ROOT)


def gen_submission(loader_submit, model, inv_mapping, savename):
    submit_csv = pd.read_csv(SAMPLE_PATH)
    submit_csv = submit_csv.set_index('id')

    results = {'id': [], 'landmarks': []}
    model.cuda()
    model.eval()

    tot = len(loader_submit)
    counts = 0
    with torch.no_grad():
        for i, sample in enumerate(loader_submit):
            x = sample['image'].to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            scores = model(x)

            # TODO: delf topk
            # probs, preds = torch.topk(softmax, NUM_TOP_PREDICTS)

            softmax = F.softmax(scores, dim=1)
            probs, preds = softmax.max(1)
            probs, preds = probs.cpu().numpy(), preds.cpu().numpy()
            # Inverse mapping before filtering minority
            preds = [inv_mapping[p] for p in preds]

            for name, pred, prob in zip(sample['name'], preds, probs):
                results['id'].append(name)
                results['landmarks'].append(f"{pred} {prob}")
                counts += 1

            if i % 50 == 0:
                now = datetime.now().strftime('%m-%d %H:%M:%S')
                print(f"[{now}] Inferring {i} / {tot} [{counts} / {submit_csv.shape[0]}] ...")

    now = datetime.now().strftime('%m%d%H%M%S')
    savename = f"{savename[:-4]}_{now}.csv"
    df = pd.DataFrame.from_dict(results)
    df = df.set_index('id')
    submit_csv.update(df)
    submit_csv.to_csv(savename)
    print(f"[{now}] Saved {savename}")
    return df


def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    return model


def submit():
    loader_submit = load_dataset_submit(root_dir=CONF.test_root, csv_file=CONF.test_file, input_size=(96, 96),
                                        batch_size=512, num_workers=4)

    inv_mapping = np.load(INV_MAPPING_PATH).item()
    num_classes = len(inv_mapping.keys())
    model = load_checkpoint(curr_model(pretrained=True, num_classes=num_classes), CHECKPOINT_PATH)

    sub_df = gen_submission(loader_submit, model, inv_mapping, savename=OUT_PATH)
    return sub_df


def do_delf(submission_path, delf_savename):
    delf = DeLF(submission_path,
                test_delf_root=TEST_DELF, train_delf_root=TRAIN_DELF,
                cls_root=CLS_ROOT, filtered_root=FILTERED_ROOT, conf_threshold=THRESHOLD)
    delf.master()
    delf.gen_filtered_submission(delf_savename)


if __name__ == "__main__":
    # sub_df = submit()
    do_delf(OUT_PATH, OUT_DELF_PATH)
