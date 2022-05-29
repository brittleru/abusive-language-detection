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
WASSEM_DIR = os.path.join(DATA_DIR, "hateful-wassem")
TRAIN_SET_PATH = os.path.join(WASSEM_DIR, "wassem_hovy_naacl.tsv")


def plot_distribution(dataframe: Union[TextFileReader, DataFrame], labels: List[str]):
    sexism_tweets = dataframe[dataframe["Label"] == "sexism"]
    racism_tweets = dataframe[dataframe["Label"] == "racism"]
    neither_tweets = dataframe[dataframe["Label"] == "none"]

    print(f"Number of sexism tweets in data set: {len(sexism_tweets)}")    # 3148
    print(f"Number of racism tweets in data set: {len(racism_tweets)}")    # 1939
    print(f"Number of neither tweets in data set: {len(neither_tweets)}")  # 11115

    labels_names = set()
    for label in labels:
        labels_names.add(label)

    plt.figure()
    labels_cap = [label_name.capitalize() for label_name in labels_names]
    labels_cap.sort()
    print(labels_cap)
    num_of_tweets_by_types = [len(neither_tweets), len(racism_tweets), len(sexism_tweets)]
    plt.pie(num_of_tweets_by_types, colors=["#E2F0CB", "#FFBB99", "#FF9999"], shadow=True, autopct="%1.1f%%",
            startangle=90)
    # "#E2F0CB"  C7CEEA  E2F0CB  FF9999
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

    print(train_df.head())  # [5 rows x 5 columns]
    print(train_df.shape)   # (16202, 5)

    train_texts = train_df["Text"].tolist()
    train_labels = train_df["Label"].tolist()

    # plot_distribution(dataframe=train_df, labels=train_labels)

    # Clean: 41462 | No lowercase: 18973 | Lowercase: 15226 | Lowercase & Stemming: 10845 | Lowercase & Lemmas: 13671
    print(f"Total number of unique words in dataset: {len(get_unique_words_vocabulary(train_texts))}")

    # Clean: 34 | No lowercase: 25 | Lowercase: 24 | Lowercase & Stemming: 24 | Lowercase & Lemmas: 24
    print(f"Longest sentence in dataset has: {get_max_words_in_tweets(train_texts)} words")
