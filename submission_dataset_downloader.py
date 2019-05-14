#!/usr/bin/python

# Based on: https://www.kaggle.com/tobwey/landmark-recognition-challenge-image-downloader

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv
import requests
from PIL import Image

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

SUBMISSION_CSV_PATH = "~/dataset/test.csv"
OUT_DIR = "~/dataset/test"

out_dir = os.path.expanduser(OUT_DIR)
data_file = os.path.expanduser(SUBMISSION_CSV_PATH)


def ParseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


def DownloadImage(key_url):
    (key, url) = key_url
    filename = os.path.join(out_dir, '%s.jpg' % key)

    if os.path.exists(filename):
        print('Image %s already exists. Skipping download.' % filename)
        return
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
    except:
        print(f"Image {key} can't be downloaded from \"{url}\"")
        return


def Run():
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = ParseData(data_file)
    pool = multiprocessing.Pool(processes=50)
    pool.map(DownloadImage, key_url_list)


if __name__ == '__main__':
    Run()
