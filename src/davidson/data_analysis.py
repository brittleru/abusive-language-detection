import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from pathlib import Path
from src.utils import process_data


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
DAVIDSON_DIR = os.path.join(DATA_DIR, "hatespeech-davidson")

DATASET_PATH = os.path.join(DAVIDSON_DIR, "data/labeled_data.csv")

df = pd.read_csv(DATASET_PATH, delimiter=",")
# df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
# df.drop(columns=["X1"], axis=1, inplace=True)
print(df.head())  # [5 rows x 7 columns]
print(df.shape)  # (24783, 7)

train_text = df["tweet"].tolist()

# for tweet in train_text:
#     print(process_data(tweet))

def get_unique_words_vocabulary(dataset: list) -> dict:
    vocabulary = {}
    for tweet in dataset:
        temp_words = tweet.split(" ")
        # temp_words = process_data(tweet, do_stemming=True, do_lemmas=False, do_lowercase=True).split(" ")
        for word in temp_words:
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)

    return vocabulary


def get_max_words_in_tweets(dataset: list) -> int:
    max_len = 0
    max_tweet = ""
    for tweet in dataset:
        temp_tweet = tweet.split(" ")
        # temp_tweet = process_data(tweet, do_stemming=True, do_lemmas=False, do_lowercase=True).split(" ")
        if max_len < len(temp_tweet):
            max_len = len(temp_tweet)
            max_tweet = temp_tweet

    print(f"Max sentence: {max_tweet}")
    return max_len


# print(get_unique_words_vocabulary(train_text))

# Clean: 59462 | No lowercase: 24667 | Lowercase: 19586 | Lowercase & Stemming: 15283 | Lowercase & Lemmas: 17819
print(f"Total number of unique words in dataset: {len(get_unique_words_vocabulary(train_text))}")

# Clean: 36 | No lowercase: 29 | Lowercase: 28 | Lowercase & Stemming: 28 | Lowercase & Lemmas: 28
print(f"Longest sentence in dataset has: {get_max_words_in_tweets(train_text)} words")
