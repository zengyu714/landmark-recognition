"""
Ref:
- https://github.com/anishagg/Google-Landmark-Recognition/blob/master/Scripts/DeLF.ipynb
- https://tfhub.dev/google/delf/1
"""
import sys

sys.path.append("/home/kimmy/models/research")

from glob import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform
from delf import feature_io
from tqdm import tqdm

from utils.util import id_to_path

_DISTANCE_THRESHOLD = 0.8
_MAXIMUM_MATCH = 20

K_NEAREST = 10


# def run_delf_module(image_path):
#     # Prepare an image tensor.
#     img_raw = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(img_raw, channels=3)
#     image = tf.image.convert_image_dtype(image, tf.float32)
#
#     # Instantiate the DELF module.
#     delf_module = hub.Module("https://tfhub.dev/google/delf/1")
#
#     delf_inputs = {
#         # An image tensor with dtype float32 and shape [height, width, 3], where
#         # height and width are positive integers:
#         'image'          : image,
#         # Scaling factors for building the image pyramid as described in the paper:
#         'image_scales'   : [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
#         # Image features whose attention score exceeds this threshold will be
#         # returned:
#         'score_threshold': 100.0,
#         # The maximum number of features that should be returned:
#         'max_feature_num': 1000,
#     }
#
#     # Apply the DELF module to the inputs to get the outputs.
#     delf_outputs = delf_module(delf_inputs, as_dict=True)
#
#     # delf_outputs is a dictionary of named tensors:
#     # * delf_outputs['locations']: a Tensor with dtype float32 and shape [None, 2],
#     #   where each entry is a coordinate (vertical-offset, horizontal-offset) in
#     #   pixels from the top-left corner of the image.
#     # * delf_outputs['descriptors']: a Tensor with dtype float32 and shape
#     #   [None, 40], where delf_outputs['descriptors'][i] is a 40-dimensional
#     #   descriptor for the image at location delf_outputs['locations'][i].
#
#     return delf_outputs['locations'], delf_outputs['descriptors']


def image_index_2_accumulated_indexes(index, accumulated_indexes_boundaries):
    """
    Image index to accumulated/aggregated locations/descriptors pair indexes.
    """
    if index > len(accumulated_indexes_boundaries) - 1:
        return None
    if index == 0:
        accumulated_index_start = 0
        accumulated_index_end = accumulated_indexes_boundaries[index]
    else:
        accumulated_index_start = accumulated_indexes_boundaries[index - 1]
        accumulated_index_end = accumulated_indexes_boundaries[index]
    return np.arange(accumulated_index_start, accumulated_index_end)


def get_locations_2_use(query_image_locations, locations_agg, image_db_index, k_nearest_indices,
                        accumulated_indexes_boundaries):
    """
    Get a pair of locations to use, the query image to the database image with given index.
    Return: a tuple of 2 numpy arrays, the locations pair.
    """
    image_accumulated_indexes = image_index_2_accumulated_indexes(image_db_index, accumulated_indexes_boundaries)
    locations_2_use_query = []
    locations_2_use_db = []
    for i, row in enumerate(k_nearest_indices):
        for acc_index in row:
            if acc_index in image_accumulated_indexes:
                locations_2_use_query.append(query_image_locations[i])
                locations_2_use_db.append(locations_agg[acc_index])
                break
    return np.array(locations_2_use_query), np.array(locations_2_use_db)


class DeLF:
    def __init__(self, submission_csv, test_delf_root, train_delf_root, cls_root, filtered_root, conf_threshold):
        """
        Write landmark id with no landmark to filtered_root
        Args:
            - submission_csv: the path of the submission csv file
            - test_delf_root: the path to all extracted testing delf
            - train_delf_root: the path to all extracted training delf
            - cls_root: the root to the txt file contains image paths have the class.
                        It can be generated by `group_landmarkid_by_class` in `utils.util`
            - filtered_root: directory name contains all filtered class (for parallel)
            - conf_threshold: when the model's output probability is less than this threshold, we do delf.
        """
        self.submission_csv = submission_csv
        assert os.path.exists(submission_csv), "No submission file to do delf"
        self.submission_df = pd.read_csv(submission_csv)

        self.test_delf_root = test_delf_root
        self.train_delf_root = train_delf_root
        self.cls_root = cls_root
        self.filtered_root = filtered_root

        self.filtered = [i for i in os.listdir(filtered_root)]
        self.conf_threshold = conf_threshold

    def get_db_paths(self, pred_cls, data_root):
        img_with_same_cls_file = glob(os.path.join(self.cls_root, f"cls_{str(pred_cls).zfill(6)}_*.txt"))
        with open(img_with_same_cls_file[0]) as f:
            lines = f.readlines()
        db_paths = [id_to_path(data_root, l.strip()) for l in lines]
        return db_paths

    def build_cls_KDtree(self, pred_cls):
        db_delfs = self.get_db_paths(pred_cls, self.train_delf_root)
        db_delfs = [i[:-4] + ".delf" for i in db_delfs[:min(_MAXIMUM_MATCH, len(db_delfs))]]

        results_dict = {'loc'       : [],
                        'descriptor': [],
                        'boundaries': []}  # Stores the locations and their descriptors for each image
        for delf_path in db_delfs:
            if not os.path.exists(delf_path):
                continue
            locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(delf_path)

            results_dict['loc'].append(locations_1)
            results_dict['descriptor'].append(descriptors_1)
            results_dict['boundaries'].append(locations_1.shape[0])

        try:
            locations_agg = np.concatenate(results_dict['loc'])
            descriptors_agg = np.concatenate(results_dict['descriptor'])
            accumulated_indexes_boundaries = np.cumsum(results_dict['boundaries'])
        except ValueError as e:
            print(e)
            return None
        # build the KD tree
        d_tree = cKDTree(descriptors_agg)
        return d_tree, (locations_agg, descriptors_agg, accumulated_indexes_boundaries)

    def match_landmark(self, query_feat_path, pred_cls):
        result = self.build_cls_KDtree(pred_cls=pred_cls)
        if result is None:
            return
        d_tree, (locations_agg, descriptors_agg, accumulated_indexes_boundaries) = result
        # query
        if not os.path.exists(query_feat_path):
            return
        query_image_locations, _, query_image_descriptors, _, _ = feature_io.ReadFromFile(query_feat_path)

        # k-nearest neighbors
        try:
            distances, indices = d_tree.query(
                    query_image_descriptors, distance_upper_bound=_DISTANCE_THRESHOLD, k=K_NEAREST, n_jobs=-1)
        except ValueError as e:
            print(e)
            return
        # Find the list of unique accumulated/aggregated indexes
        unique_indices = np.sort(np.unique(indices))
        if unique_indices[-1] == descriptors_agg.shape[0]:
            unique_indices = unique_indices[:-1]

        unique_image_indexes = np.unique([np.argmax(np.array(accumulated_indexes_boundaries) > index)
                                          for index in unique_indices])

        # Array to keep track of all candidates in database.
        inliers_counts = []
        for index in unique_image_indexes:
            locations_2_use_query, locations_2_use_db = \
                get_locations_2_use(query_image_locations, locations_agg,
                                    index, indices, accumulated_indexes_boundaries)
            # Perform geometric verification using RANSAC.
            _, inliers = ransac(
                    (locations_2_use_db, locations_2_use_query),  # source and destination coordinates
                    AffineTransform,
                    min_samples=3,
                    residual_threshold=20,
                    max_trials=1000)
            # If no inlier is found for a database candidate image, we continue on to the next one.
            if inliers is None or len(inliers) == 0:
                continue
            # the number of inliers as the score for retrieved images.
            inliers_counts.append({"index": index, "inliers": sum(inliers)})

        inliers_list = []
        for inl in inliers_counts:
            inliers_list.append(inl['inliers'])
        return inliers_list

    def worker(self, img_id, score, i, tot_num):
        if i % 50 == 0:
            now = datetime.now().strftime('%m-%d %H:%M:%S')
            print(f"[{now}] Processing {i} / {tot_num}...")
        pred, conf = score.split()
        pred, conf = int(pred), float(conf)
        # skip confident ones
        if conf > self.conf_threshold:
            pass

        query_feat_path = id_to_path(self.test_delf_root, img_id)[:-4] + ".delf"
        all_inliers = self.match_landmark(query_feat_path, pred)
        if all_inliers is None:  # didn't have corresponding test delf feature
            pass

        if all_inliers and np.mean(all_inliers) < 5:
            # print(f"No landmark in {query_feat_path.split('/')[-1][:-5]}")
            snapshot = os.path.join(self.filtered_root, f"{img_id}")
            if not os.path.exists(snapshot):
                os.mknod(snapshot)

    def master(self):
        # resume
        df = self.submission_df
        start = df[df['id'].isin(self.filtered)].index.max()
        df = df if start is np.nan else df.iloc[start:]
        Parallel(n_jobs=-1, verbose=1)(
                delayed(self.worker)(img_id, score, i, len(df)) for i, (img_id, score) in df.iterrows())


