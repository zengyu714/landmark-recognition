import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import load_dataset_submit
from models.resnet import resnet50
from models.se_resnet import se_resnet50

parser = argparse.ArgumentParser(description='Google Landmark Recognition Challenge')
parser.add_argument('-g', '--cuda-device', type=int, default=0,
                    help='Choose which gpu to use (default: 0)')

args = parser.parse_args()
if torch.cuda.is_available():
    torch.cuda.set_device(args.cuda_device)
    device = torch.cuda.current_device()
else:
    device = torch.device('cpu')

INV_MAPPING_PATH = "/home/kimmy/dataset/mapping_inv.npy"
CHECKPOINT_ROOT = "/home/kimmy/landmark-recognition/checkpoints/"
CHECKPOINT_NAME = "finetune_all_best.ckpt"
SUBMIT_PATH = "/home/kimmy/landmark-recognition/submission/submission.csv"
SAMPLE_PATH = "/home/kimmy/dataset/recognition_sample_submission.csv"


def gen_submission(loader_submit, model, inv_mapping, submit_path):
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

            if i % 10 == 0:
                now = datetime.now().strftime('%m-%d %H:%M:%S')
                print(f"[{now}] Inferring {i} / {tot} [{counts} / {submit_csv.shape[0]}] ...")

        df = pd.DataFrame.from_dict(results)
        df = df.set_index('id')
        submit_csv.update(df)
        submit_csv.to_csv(submit_path)
        print(f"[{now}] Saved {submit_path}")


def load_checkpont(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    return model


if __name__ == "__main__":
    loader_submit = load_dataset_submit(batch_size=512)

    inv_mapping = np.load(INV_MAPPING_PATH).item()
    num_classes = len(inv_mapping.keys())
    model = load_checkpont(resnet50(pretrained=False, num_classes=num_classes), CHECKPOINT_ROOT + CHECKPOINT_NAME)

    gen_submission(loader_submit, model, inv_mapping, submit_path=SUBMIT_PATH)
