import os
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from pandas import DataFrame
from typing import List, Dict, Union
from pandas.io.parsers import TextFileReader

from src.utils import process_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
HATE_CHECK_DIR = os.path.join(DATA_DIR, "hatecheck-data")
DATASET_PATH = os.path.join(HATE_CHECK_DIR, "test_suite_cases.csv")


def plot_distribution(dataframe: Union[TextFileReader, DataFrame], labels: List[str]):
    not_hateful = dataframe[dataframe["label_gold"] == "non-hateful"]
    hateful = dataframe[dataframe["label_gold"] == "hateful"]

    print(f"Number of not-hateful texts in data set: {len(not_hateful)}")    # 1165
    print(f"Number of hateful texts in data set: {len(hateful)}")            # 2563

    labels_names = set()
    for label in labels:
        labels_names.add(label)

    plt.figure()
    labels_cap = [label_name.capitalize() for label_name in labels_names]
    labels_cap.sort()
    print(labels_cap)
    num_of_tweets_by_types = [len(hateful), len(not_hateful)]
    plt.pie(num_of_tweets_by_types, colors=["#FF9999", "#E2F0CB"], shadow=True, autopct="%1.1f%%",
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
    df = pd.read_csv(DATASET_PATH, delimiter=",")

    print(df.head())  # [5 rows x 13 columns]
    print(df.shape)   # (3728, 13)

    train_texts = df["test_case"].tolist()
    train_labels = df["label_gold"].tolist()

    # plot_distribution(dataframe=df, labels=train_labels)

    # Clean: 1828 | No lowercase: 1316 | Lowercase: 1179 | Lowercase & Stemming: 1004 | Lowercase & Lemmas: 1114
    print(f"Total number of unique words in dataset: {len(get_unique_words_vocabulary(train_texts))}")

    # Clean: 20 | No lowercase: 12 | Lowercase: 11 | Lowercase & Stemming: 11 | Lowercase & Lemmas: 11
    print(f"Longest sentence in dataset has: {get_max_words_in_tweets(train_texts)} words")
