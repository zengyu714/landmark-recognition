import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import load_dataset_submit
from models.resnet import resnet50
from utils.delf.match_images import delf_master
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

CHECKPOINT_PATH = os.path.join(CONF.ckpt_root, "finetune_all_best.ckpt")
INV_MAPPING_PATH = os.path.join(CONF.data_root, "mapping_inv.npy")
SAMPLE_PATH = os.path.join(CONF.data_root, "recognition_sample_submission.csv")

TEST_DELF = "/home/kimmy/dataset-96/test_delf"
CLS_ROOT = "/home/kimmy/dataset-96/cls"

OUT_PATH = os.path.join(CONF.submit_root, "submission.csv")


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

    df = pd.DataFrame.from_dict(results)
    df = df.set_index('id')
    submit_csv.update(df)
    submit_csv.to_csv(savename)
    print(f"[{now}] Saved {savename}")
    return df


def do_delf(submission_df, delf_savename, test_delf, cls_root):
    delf_csv = delf_master(submission_df, test_delf, cls_root)
    delf_csv.to_csv(delf_savename)
    print(f"Saved {delf_savename} filtered by delf")


def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    return model


def submit(only_delf=False):
    if only_delf:
        assert os.path.exists(OUT_PATH), "No submission file to do delf"
        sub_df = pd.read_csv(OUT_PATH)
        do_delf(sub_df, f"{OUT_PATH[:-4]}_delf.csv", TEST_DELF, CLS_ROOT)
    else:
        loader_submit = load_dataset_submit(batch_size=512)

        inv_mapping = np.load(INV_MAPPING_PATH).item()
        num_classes = len(inv_mapping.keys())
        model = load_checkpoint(resnet50(pretrained=False, num_classes=num_classes), CHECKPOINT_PATH)

        sub_df = gen_submission(loader_submit, model, inv_mapping, savename=OUT_PATH)
        do_delf(sub_df, f"{OUT_PATH[:-4]}_delf.csv", TEST_DELF, CLS_ROOT)


if __name__ == "__main__":
    submit(only_delf=True)
