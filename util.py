from functools import partial
import numpy as np
import os
import visdom


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
