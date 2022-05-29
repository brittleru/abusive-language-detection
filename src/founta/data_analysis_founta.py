import os
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from pathlib import Path
from src.utils import process_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
FOUNTA_DIR = os.path.join(DATA_DIR, "large-founta")

TRAIN_SET_PATH = os.path.join(FOUNTA_DIR, "train.tsv")
VAL_SET_PATH = os.path.join(FOUNTA_DIR, "dev.tsv")
TEST_SET_PATH = os.path.join(FOUNTA_DIR, "test.tsv")


def plot_distribution(dataframe, labels: List[str]):
    spam_tweets = dataframe[dataframe["class"] == "spam"]
    normal_tweets = dataframe[dataframe["class"] == "normal"]
    hateful_tweets = dataframe[dataframe["class"] == "hateful"]
    abusive_tweets = dataframe[dataframe["class"] == "abusive"]

    print(f"Number of spam tweets in data set: {len(spam_tweets)}")          # 12873
    print(f"Number of normal tweets in data set: {len(normal_tweets)}")      # 51842
    print(f"Number of hateful tweets in data set: {len(hateful_tweets)}")    # 4897
    print(f"Number of abusive tweets in data set: {len(abusive_tweets)}\n")  # 26672

    labels_names = set()
    for label in labels:
        labels_names.add(label)

    plt.figure()
    labels_cap = [label_name.capitalize() for label_name in labels_names]
    labels_cap.sort()
    num_of_tweets_by_types = [len(abusive_tweets), len(hateful_tweets), len(normal_tweets), len(spam_tweets)]
    plt.pie(num_of_tweets_by_types, colors=["#FF9999", "#FFBB99", "#C7CEEA", "#E2F0CB"], shadow=True, autopct="%1.1f%%",
            startangle=90)
    plt.legend(labels_cap)
    plt.axis("equal")
    plt.show()


def get_unique_words_vocabulary(dataset: List[str]) -> Dict[str, int]:
    vocabulary = {}
    for tweet in dataset:
        # temp_words = tweet.split(" ")
        temp_words = process_data(tweet, do_stemming=False, do_lemmas=True, do_lowercase=True).split(" ")
        for word in temp_words:
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)

    return vocabulary


def get_max_words_in_tweets(dataset: List[str]) -> int:
    max_len = 0
    min_len = 9999999
    max_tweet = ""
    min_tweet = ""
    for tweet in dataset:
        # temp_tweet = tweet.split(" ")
        temp_tweet = process_data(tweet, do_stemming=False, do_lemmas=True, do_lowercase=True).split(" ")
        if max_len < len(temp_tweet):
            max_len = len(temp_tweet)
            max_tweet = temp_tweet
        if min_len > len(temp_tweet):
            min_len = len(temp_tweet)
            min_tweet = temp_tweet

    print(f"Min sentance: {min_tweet}")
    print(f"Max sentence: {max_tweet}")
    return max_len


if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_SET_PATH, sep="\t", header=0)
    val_df = pd.read_csv(VAL_SET_PATH, sep="\t", header=0)
    test_df = pd.read_csv(TEST_SET_PATH, sep="\t", header=0)

    print(train_df.head())  # [5 rows x 3 columns]
    print(val_df.head())    # [5 rows x 3 columns]
    print(test_df.head())   # [5 rows x 3 columns]

    print(train_df.shape)  # (69322, 3)
    print(val_df.shape)    # (7703,  3)
    print(test_df.shape)   # (19259, 3)

    train_texts = train_df["sentence"].tolist()
    val_texts = val_df["sentence"].tolist()
    test_texts = test_df["sentence"].tolist()

    dataset_df = pd.concat([train_df, val_df, test_df])
    print(dataset_df.head())  # [5 rows x 3 columns]
    print(dataset_df.shape)   # (96284, 3)

    class_labels = dataset_df["class"].tolist()
    dataset_texts = dataset_df["sentence"].tolist()

    print("\n")

    # Clean: 292512 | No lowercase: 97205 | Lowercase: 71500 | Lowercase & Stemming: 56824 | Lowercase & Lemmas: 66265
    print(f"Total number of unique words in dataset: {len(get_unique_words_vocabulary(dataset_texts))}")

    # Clean: 48 | No lowercase: 36 | Lowercase: 30 | Lowercase & Stemming: 30 | Lowercase & Lemmas: 30
    print(f"Longest sentence in dataset has: {get_max_words_in_tweets(dataset_texts)} words")

    plot_distribution(dataframe=dataset_df, labels=class_labels)
    # print("\n\n")
    # plot_distribution(dataframe=test_df, labels=test_df["class"].tolist())
