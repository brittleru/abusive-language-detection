import os

import numpy as np
from pathlib import Path
from tqdm import tqdm
import string

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
PRETRAINED_DIR = os.path.join(DATA_DIR, "pretrained")
GLOVE_PATH_6B = os.path.join(PRETRAINED_DIR, "glove-6B")
GLOVE_PATH_840B = os.path.join(PRETRAINED_DIR, "glove-840B")


def create_embeddings(path: str) -> dict:
    embeddings_index = {}

    f = open(path, "r", errors='ignore', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefficients
    f.close()

    return embeddings_index


# print('Found %s word vectors.' % len(create_embeddings(os.path.join(GLOVE_PATH_6B, "glove.6B.50d.txt"))))


# print('Found %s word vectors.' % create_embeddings(os.path.join(GLOVE_PATH_6B, "glove.6B.50d.txt")))
# print('Found %s word vectors.' % len(create_embeddings(os.path.join(GLOVE_PATH_6B, "glove.6B.100d.txt"))))
# print('Found %s word vectors.' % len(create_embeddings(os.path.join(GLOVE_PATH_6B, "glove.6B.200d.txt"))))
# print('Found %s word vectors.' % len(create_embeddings(os.path.join(GLOVE_PATH_6B, "glove.6B.300d.txt"))))
# print('Found %s word vectors.' % len(create_embeddings(os.path.join(GLOVE_PATH_840B, "glove.840B.300d.txt"))))

def get_dimension_size(line):
    size = 0
    l_split = line.strip().split()
    for i in l_split:
        try:
            _ = float(i)
            size = size + 1
        except:
            pass
    return size


def get_embeddings(file):
    embs = dict()
    firstLine = open(file, 'r').readline()
    dimension = get_dimension_size(firstLine)  # look at the first line to get the dimension
    for l in open(file, 'rb').readlines():
        l_split = l.decode('utf8').strip().split()
        if len(l_split) == 2:
            continue
        emb = l_split[-1 * dimension:]  # use the dimension to mark the boundary
        word = l_split[:-1 * dimension]
        word = ''.join(word)
        embs[word] = [float(em) for em in emb]
        # print("Got {} embeddings from {}".format(len(embs), file))
    return embs


# print('Found %s word vectors.' % len(get_embeddings(os.path.join(GLOVE_PATH_840B, "glove.840B.300d.txt"))))
