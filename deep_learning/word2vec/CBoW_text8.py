#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-25 下午9:19
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : CBoW_text8.py
# @Software: PyCharm



import collections
import math
import os
import random
import zipfile

from config import BASE_PATH
import numpy as np
from six.moves import urllib
from six.moves import xrange
from deep_learning.word2vec.CBoW import CBOW
from deep_learning.manual_neural_network.FC_classification_v2 import BackPropagation

# Step1: prepare data.
url = 'http://mattmahoney.net/dc/'

def download(filename):
    print('Starting downloading data ...')
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
    print('Data downloaded!')
    return filename


def maybe_download(filename, expected_bytes):
    """Download a file if not present or size is incorrect."""
    if not os.path.exists(filename):
        filename = download(filename)
    else:
        if os.stat(filename).st_size != expected_bytes:
            os.remove(filename)
            filename = download(filename)

    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename



# Step2: split words.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

if __name__ == "__main__":
    np.random.seed(1)

    # 1、语料
    corpus_file_name = os.path.join(BASE_PATH, 'data_on_git/text8.zip')
    filename = maybe_download(corpus_file_name, 31344016)
    print("Data prepared!")
    words = read_data(filename)
    print('Data size', len(words))

    # text_words，剔除低频词后的文章
    cbow = CBOW(text_words=words[:100000])
    cbow.train(iter=10000)

    cbow.get_similar(b'four')
