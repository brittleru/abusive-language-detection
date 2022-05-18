import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from src.utils import process_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDGEN_DIR = os.path.join(DATA_DIR, "dynamically-hate-vidgen")

DATASET_PATH = os.path.join(VIDGEN_DIR, "Dynamically_Generated_Hate_Dataset_v0.2.3.csv")

df = pd.read_csv(DATASET_PATH, delimiter=",")
# df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
# df.drop(columns=["X1"], axis=1, inplace=True)
print(df.head())  # [5 rows x 13 columns]
print(df.shape)  # (41144, 13)

# Number of different type values
print(df.groupby("type").count()["acl.id"])

train_text = df["text"].tolist()


def get_unique_words_vocabulary(dataset: list) -> dict:
    vocabulary = {}
    for tweet in dataset:
        temp_words = process_data(tweet, do_stemming=False, do_lemmas=True, do_lowercase=True).split(" ")
        for word in temp_words:
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)

    return vocabulary


def get_max_words_in_tweets(dataset: list) -> int:
    max_len = 0
    for tweet in dataset:
        temp_tweet = process_data(tweet, do_stemming=False, do_lemmas=True, do_lowercase=True).split(" ")
        if max_len < len(temp_tweet):
            max_len = len(temp_tweet)

    return max_len


# print(get_unique_words_vocabulary(train_text))

# Clean: 57001 | No lowercase: 31643 | Lowercase: 26095 | Lowercase & Stemming: 17709 | Lowercase & Lemmas: 23102
print(f"Total number of unique words in dataset: {len(get_unique_words_vocabulary(train_text))}")

# Clean: 408 | No lowercase: 241 | Lowercase: 218 | Lowercase & Stemming: 218 | Lowercase & Lemmas: 218
print(f"Longest sentence in dataset has: {get_max_words_in_tweets(train_text)} words")
