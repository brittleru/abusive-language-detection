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
ZAMPIERI_DIR = os.path.join(DATA_DIR, "olid-zampieri")
TRAIN_SET_PATH = os.path.join(ZAMPIERI_DIR, "cleaned_train_data_v1.csv")
TEST_SET_PATH = os.path.join(ZAMPIERI_DIR, "cleaned_test_data_v1.csv")


def plot_distribution(dataframe: Union[TextFileReader, DataFrame], labels: List[str]):
    not_offensive_tweets = dataframe[dataframe["subtask_a"] == "NOT"]
    offensive_tweets = dataframe[dataframe["subtask_a"] == "OFF"]

    print(f"Number of not offensive tweets in data set: {len(not_offensive_tweets)}")    # 9460
    print(f"Number of offensive tweets in data set: {len(offensive_tweets)}")            # 4640

    labels_names = set()
    for label in labels:
        labels_names.add(label)

    plt.figure()
    labels_cap = [label_name.capitalize() for label_name in labels_names]
    labels_cap.sort()
    print(labels_cap)
    num_of_tweets_by_types = [len(not_offensive_tweets), len(offensive_tweets)]
    plt.pie(num_of_tweets_by_types, colors=["#E2F0CB", "#FF9999"], shadow=True, autopct="%1.1f%%",
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
    train_df = pd.read_csv(TRAIN_SET_PATH, delimiter=",")
    test_df = pd.read_csv(TEST_SET_PATH, delimiter=",")

    print(train_df.head())  # [5 rows x 7 columns]
    print(train_df.shape)   # (16202, 5)
    print(test_df.head())   # [5 rows x 5 columns]
    print(test_df.shape)    # (860, 5)

    dataset_df = pd.concat([train_df, test_df])

    print(dataset_df.head())  # [5 rows x 7 columns]
    print(dataset_df.shape)   # (14100, 7)

    train_texts = train_df["tweet"].tolist()
    train_labels = train_df["subtask_a"].tolist()
    test_texts = test_df["tweet"].tolist()
    test_labels = test_df["subtask_a"].tolist()

    dataset_texts = dataset_df["tweet"].tolist()
    dataset_labels = dataset_df["subtask_a"].tolist()

    # plot_distribution(dataframe=dataset_df, labels=dataset_labels)

    # Clean: 42731 | No lowercase: 25045 | Lowercase: 19449 | Lowercase & Stemming: 14300 | Lowercase & Lemmas: 17613
    print(f"Total number of unique words in dataset: {len(get_unique_words_vocabulary(dataset_texts))}")

    # Clean: 103 | No lowercase: 60 | Lowercase: 42 | Lowercase & Stemming: 42 | Lowercase & Lemmas: 42
    print(f"Longest sentence in dataset has: {get_max_words_in_tweets(dataset_texts)} words")
