# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Extracts DELF features from a list of images, saving them to file.

The images must be in JPG format. The program checks if descriptors already
exist, and skips computation for those.
"""
import sys

sys.path.append("/home/kimmy/models/research")  # to import `object_detection`
sys.path.append("../..")  # to import configuration

import os

import argparse
import sys
import time

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app

from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io
from glob import glob
from config import Config

CONF = Config(data_root="/home/kimmy/dataset-96")

cmd_args = None

# Extension of feature files.
_DELF_EXT = '.delf'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100
# For parallel processing
_GROUP_NUMS = 2

_DO_THRESHOLD = True
_THRE_IMAGE_LIST = os.path.join(CONF.data_root, "cls_list.txt")

sess_config = tf.ConfigProto(allow_soft_placement=True)


def _ReadImageList(curr_idx):
    """Helper function to read image paths from data root.

    Args:
      data_root: Path to all images
      curr_idx: 0, 1, 2..., GROUP_NUMS - 1

    Returns:
      image_paths: List of image paths.
    """
    assert curr_idx < _GROUP_NUMS, f"Max index should be {_GROUP_NUMS - 1}"

    image_paths = glob(os.path.join(cmd_args.img_root, "**/*.jpg"), recursive=True)
    exist_delfs = glob(os.path.join(cmd_args.output_dir, "**/*.delf"), recursive=True)
    exist_delfs = [i.replace('.delf', '.jpg').replace(cmd_args.output_dir, cmd_args.img_root) for i in exist_delfs]
    image_paths = list(set(image_paths).difference(set(exist_delfs)))

    if "train" in cmd_args.img_root and _DO_THRESHOLD:
        with open(_THRE_IMAGE_LIST) as f:
            lines = f.readlines()
        limited = [os.path.join(cmd_args.img_root, '/'.join(l[:3]), l.rstrip()) for l in lines]
        image_paths = list(set(limited).intersection(set(image_paths)))  # wired error otherwise

    tot_nums = len(image_paths)
    size = int(tot_nums // _GROUP_NUMS)
    if curr_idx == _GROUP_NUMS - 1:
        return image_paths[size * curr_idx:]
    return image_paths[size * curr_idx: size * (curr_idx + 1)]


def MakeExtractor(sess, config, import_scope=None):
    """Creates a function to extract features from an image.

    Args:
      sess: TensorFlow session to use.
      config: DelfConfig proto containing the model configuration.
      import_scope: Optional scope to use for model.

    Returns:
      Function that receives an image and returns features.
    """
    tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING],
            config.model_path,
            import_scope=import_scope)
    import_scope_prefix = import_scope + '/' if import_scope is not None else ''
    input_image = sess.graph.get_tensor_by_name('%sinput_image:0' % import_scope_prefix)
    input_score_threshold = sess.graph.get_tensor_by_name('%sinput_abs_thres:0' % import_scope_prefix)
    input_image_scales = sess.graph.get_tensor_by_name('%sinput_scales:0' % import_scope_prefix)
    input_max_feature_num = sess.graph.get_tensor_by_name('%sinput_max_feature_num:0' % import_scope_prefix)
    boxes = sess.graph.get_tensor_by_name('%sboxes:0' % import_scope_prefix)
    raw_descriptors = sess.graph.get_tensor_by_name('%sfeatures:0' % import_scope_prefix)
    feature_scales = sess.graph.get_tensor_by_name('%sscales:0' % import_scope_prefix)
    attention_with_extra_dim = sess.graph.get_tensor_by_name('%sscores:0' % import_scope_prefix)
    attention = tf.reshape(attention_with_extra_dim, [tf.shape(attention_with_extra_dim)[0]])

    locations, descriptors = feature_extractor.DelfFeaturePostProcessing(boxes, raw_descriptors, config)

    def ExtractorFn(image):
        """Receives an image and returns DELF features.

        Args:
          image: Uint8 array with shape (height, width 3) containing the RGB image.

        Returns:
          Tuple (locations, descriptors, feature_scales, attention)
        """
        return sess.run(
                [locations, descriptors, feature_scales, attention],
                feed_dict={
                    input_image          : image,
                    input_score_threshold: config.delf_local_config.score_threshold,
                    input_image_scales   : list(config.image_scales),
                    input_max_feature_num: config.delf_local_config.max_feature_num
                })

    return ExtractorFn


def main(unused_argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.cuda_device

    tf.logging.set_verbosity(tf.logging.INFO)

    # Read list of images.
    tf.logging.info('Reading list of images...')
    image_paths = _ReadImageList(cmd_args.curr_idx)
    num_images = len(image_paths)
    tf.logging.info('done! Found %d images', num_images)

    # Parse DelfConfig proto.
    config = delf_config_pb2.DelfConfig()
    with tf.gfile.FastGFile(cmd_args.config_path, 'r') as f:
        text_format.Merge(f.read(), config)

    # Create output directory if necessary.
    if not os.path.exists(cmd_args.output_dir):
        os.makedirs(cmd_args.output_dir, exist_ok=True)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Reading list of images.
        filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)
        image_tf = tf.image.decode_image(value, channels=3)

        count = 0
        with tf.Session(config=sess_config) as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            extractor_fn = MakeExtractor(sess, config)

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            start = time.clock()
            for i in range(num_images):
                # Write to log-info once in a while.
                if i == 0:
                    tf.logging.info('Starting to extract DELF features from images...')
                elif i % _STATUS_CHECK_ITERATIONS == 0:
                    elapsed = (time.clock() - start)
                    tf.logging.info(
                            'Processing image %d out of %d, last %d '
                            'images took %f seconds', i, num_images, _STATUS_CHECK_ITERATIONS,
                            elapsed)
                    start = time.clock()

                try:
                    im = sess.run(image_tf)
                except tf.errors.NotFoundError as e:
                    count += 1
                    tf.logging.warning(f"{'*' * 15} Decode error {e} - {count}")
                    continue

                if len(im.shape) == 4 or im.shape[-1] != 3:
                    tf.logging.warning(f"{'*' * 15} Skipping image {i} which probably is a GIF or broken image")
                    continue

                # If descriptor already exists, skip its computation.
                out_desc_filename = os.path.splitext(os.path.basename(image_paths[i]))[0] + _DELF_EXT
                out_desc_parent_dir = os.path.join(cmd_args.output_dir, '/'.join(out_desc_filename[:3]))
                if not os.path.exists(out_desc_parent_dir):
                    os.makedirs(out_desc_parent_dir)

                out_desc_fullpath = os.path.join(out_desc_parent_dir, out_desc_filename)
                if tf.gfile.Exists(out_desc_fullpath):
                    tf.logging.info('Skipping %s', image_paths[i])
                    continue

                # Extract and save features.
                (locations_out, descriptors_out, feature_scales_out,
                 attention_out) = extractor_fn(im)

                feature_io.WriteToFile(out_desc_fullpath, locations_out,
                                       feature_scales_out, descriptors_out,
                                       attention_out)

            # Finalize enqueue threads.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument(
            '--config_path',
            type=str,
            default='config.pbtxt',
            help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
    parser.add_argument(
            '--output_dir',
            type=str,
            # default=f'{CONF.train_delf}',
            default=f'{CONF.test_delf}',
            help="""
      Directory where DELF features will be written to. Each image's features
      will be written to a file with same name, and extension replaced by .delf.
      """)
    parser.add_argument(
            '--img_root',
            type=str,
            # default=f'{CONF.train_root}',
            default=f'{CONF.test_root}',
            help="""
      Directory where DELF features will be extracted, i.e., the root path 
      to the images.
      """)
    parser.add_argument(
            '--cls_root',
            type=str,
            default=f'{CONF.cls_root}',
            help="""
    The root path to the class.
    """)
    parser.add_argument(
            '--curr_idx',
            type=int,
            default=0,
            help="""
        The index of the group for the parallel processing
        """)
    parser.add_argument(
            '--cuda_device',
            type=str,
            default="0",
            help="""
        The index of the GPU
        """)
    cmd_args, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
