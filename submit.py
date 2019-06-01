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
input_size = (224, 224)

CHECKPOINT_PATH = os.path.join(CONF.ckpt_root, "seatt_111_best.ckpt")
INV_MAPPING_PATH = os.path.join(CONF.data_root, "mapping_inv.npy")
SAMPLE_PATH = os.path.join(CONF.data_root, "recognition_sample_submission.csv")

TRAIN_DELF = CONF.train_delf
TEST_DELF = CONF.test_delf
CLS_ROOT = CONF.cls_root

OUT_PATH = os.path.join(CONF.submit_root, "submission_senet.csv")
OUT_DELF_PATH = f"{OUT_PATH[:-4]}_delf.csv"
THRESHOLD = 0.2

FILTERED_ROOT = os.path.join(CONF.data_root, "filtered_senet_0.3")
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
    loader_submit = load_dataset_submit(root_dir=CONF.test_root, csv_file=CONF.test_file, input_size=input_size,
                                        batch_size=512, num_workers=8)

    inv_mapping = np.load(INV_MAPPING_PATH, allow_pickle=True).item()
    num_classes = len(inv_mapping.keys())
    model = load_checkpoint(curr_model(pretrained=True, num_classes=num_classes), CHECKPOINT_PATH)

    sub_df = gen_submission(loader_submit, model, inv_mapping, savename=OUT_PATH)
    return sub_df


def do_delf(submission_path):
    delf = DeLF(submission_path,
                test_delf_root=TEST_DELF, train_delf_root=TRAIN_DELF,
                cls_root=CLS_ROOT, filtered_root=FILTERED_ROOT, conf_threshold=THRESHOLD)
    delf.master()


def do_filter(filtered_root, subsimmsion_path, delf_savename):
    print("Processing filter")
    sub_df = pd.read_csv(subsimmsion_path)
    filtered = os.listdir(filtered_root)
    delf_df = sub_df.copy()
    delf_df.loc[delf_df['id'].isin(filtered), 'landmarks'] = ""
    delf_df.set_index('id', inplace=True)
    assert len(delf_df.columns) == 1, "Make sure we only have two columns in total for correct scoring"
    delf_df.to_csv(delf_savename)
    print(f"Saved {delf_savename} filtered by delf")


if __name__ == "__main__":
    # sub_df = submit()
    do_delf(OUT_PATH)
    # do_filter(filtered_root="/home/kimmy/dataset-96/filtered_0.2",
    #           subsimmsion_path="/home/kimmy/landmark-recognition/submission/submission_senet.csv",
    #           delf_savename="/home/kimmy/landmark-recognition/submission/submission_senet_delf.csv")
