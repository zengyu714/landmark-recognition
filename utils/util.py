import os

import numpy as np
import pandas as pd
from torch.utils import model_zoo
from torch import nn
from tqdm import tqdm


def parse_info(df, idx, root):
    """
    Parse information from dataframe
    """
    df_row = df.iloc[idx]

    img_id = str(df_row['id'])
    img_name = id_to_path(root, img_id)
    landmark_id = df_row.landmark_id
    return img_name, landmark_id


def print_basic_params(landmark):
    print(f"*** Start training {landmark.nickname} for {landmark.tot_epochs} epochs...")
    print(f"--- Current GPU device is {landmark.device}")
    print(f"--- Optimizer is {landmark.optimizer}")
    print(f"--- Batch size is {landmark.batch_size}")
    print(f"--- Parameters: {sum(p.numel() for p in landmark.model.parameters() if p.requires_grad)}")
    print(f"--- Load pretrained weights from ImageNet: {bool(landmark.pretrained)}")
    print(f"--- Use stage finetune strategy: {bool(landmark.use_stage)}")


def gap_accuracy(pred, prob, true, return_df):
    """
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition.
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictionsls
        prob: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_df: also return the data frame used in the calculation

    Returns:
        GAP score

    Ref:
        https://www.kaggle.com/davidthaler/gap-metric

    """
    x = pd.DataFrame({'pred': pred, 'conf': prob, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_df:
        return gap, x
    else:
        return gap


def unfreeze_resnet50_bottom(landmark):
    # TODO: may tune learning rate
    for i, child in enumerate(landmark.model.children()):
        # add two more modeuls to train
        if i in [6, 7]:  # conv4_x and conv5_x
            try:
                landmark.optimizer.add_param_group({'params': child.parameters()})
            except ValueError:
                print(f"Layer {i} is already exists")
                continue


def load_pretrained_weights(model, weight_url, exclude_layers=None):
    pretrained_dict = model_zoo.load_url(weight_url)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    if exclude_layers:
        for k in exclude_layers:
            pretrained_dict.pop(k)
    # print(f"Restore parameters: {pretrained_dict.keys()}")

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    return model


def load_and_modify_pretrained_num_classes(model, model_url, new_num_classes):
    model = load_pretrained_weights(model, model_url)
    if any(i in model_url for i in ['senet', 'se_res']):
        num_features = model.last_linear.in_features
        model.last_linear = nn.Linear(num_features, new_num_classes)
    else:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, new_num_classes)
    return model


def group_landmarkid_by_class(img_root, out_cls_root, train_csv):
    train_df = pd.read_csv(train_csv)
    train_df.drop(columns='url', inplace=True)

    groups = train_df.groupby(['landmark_id'])
    for landmark_id, df_group in tqdm(groups):
        # print(f"Processing landmark id {landmark_id}...")
        img_ids = df_group['id']
        img_names = [id + '.jpg\n' for id in img_ids]
        cls_str = str(landmark_id).zfill(6)
        cls_num = str(len(img_names)).zfill(5)
        with open(os.path.join(out_cls_root, f"cls_{cls_str}_#_{cls_num}.txt"), 'w+') as f:
            f.writelines(img_names)


def id_to_path(img_root, img_id):
    if not img_id.endswith(".jpg"):
        img_id += ".jpg"
    out_dir = os.path.join(img_root, '/'.join(img_id[:3]), img_id)
    return out_dir
