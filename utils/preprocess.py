"""
Resize the images to smaller size by parallel processing.
"""
from glob import glob
import os
import shutil

from joblib import Parallel, delayed
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm
from utils.util import id_to_path

SIZE = (96, 96)


def resize_worker(img_path, savename):
    if os.path.exists(savename):
        return
    try:
        img = imread(img_path)
    except:
        print(f"Cannot open image {img_path}")
        return
    resized = resize(img, SIZE, anti_aliasing=True)
    dirname = os.path.dirname(savename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    imsave(savename, resized)


def final_check(img_paths, out_paths):
    for img_path, savename in tqdm(zip(img_paths, out_paths), total=len(img_paths)):
        if os.path.exists(savename):
            continue
        resize_worker(img_path, savename)


def standardize_image_paths(img_path):
    """
        Make the training images same as the official path with
        training_root/{image_name[0]}/{image_name[1]}/{image_name[2]}/{image_name}.jpg
        E.g., image_name is `9a14d551fc6cd7a7.jpg` and stored at `train/9a14d551fc6cd7a7.jpg`
              we would distribute it to the `train/9/a/1/9a14d551fc6cd7a7.jpg`
    """
    img_dir = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    out_dir = os.path.join(img_dir, '/'.join(img_name[:3]))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, img_name)

    return out_path


def resize_master(in_dir, out_dir):
    img_paths = glob(os.path.join(in_dir, "**/*.jpg"), recursive=True)
    out_paths = [p.replace(in_dir, out_dir) for p in img_paths]
    # out_paths = [standardize_image_paths(p) for p in out_paths]

    try:
        Parallel(n_jobs=-1, verbose=1)(
                delayed(resize_worker)(img_path, savename)
                for img_path, savename in tqdm(zip(img_paths, out_paths), total=len(img_paths)))
    finally:
        # check resize
        final_check(img_paths, out_paths)


if __name__ == "__main__":
    resize_master(in_dir="/home/kimmy/dataset-256/test", out_dir="/home/kimmy/dataset-96/test")
    # resize_master(in_dir="/home/kimmy/dataset-256/train", out_dir="/home/kimmy/dataset-96/train")
