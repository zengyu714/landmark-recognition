import os

import numpy as np
import pandas as pd


def parse_info(df, idx, root):
    """
    Parse information from dataframe
    """
    df_row = df.iloc[idx]

    img_id = str(df_row['id'])
    img_name = os.path.join(root, '/'.join(img_id[:3]), img_id + '.jpg')
    landmark_id = df_row.landmark_id
    return img_name, landmark_id



def print_basic_params(landmark):
    print(f"*** Start training {landmark.modelname} for {landmark.tot_epochs} epochs...")
    print(f"- Current GPU device is {landmark.device}")
    print(f"- Optimizer is {landmark.optimizer}")
    print(f"- Batch size is {landmark.batch_size}")
    print(f"- Load pretrained weights from ImageNet: {bool(landmark.pretrained)}")
    print(f"- Use stage finetune strategy: {bool(landmark.use_stage)}")


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

