import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
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


def plots():
    #  Number of tweets depending of how are split
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    dev_df = df[df["split"] == "dev"]
    print(f"Number of train split sentences in data set: {len(train_df)}")  # 32924
    print(f"Number of test split sentences in data set: {len(test_df)}")  # 4120
    print(f"Number of dev split sentences in data set: {len(dev_df)}\n")  # 4100

    splits = np.array([len(train_df), len(test_df), len(dev_df)])
    labels = ["Train", "Test", "Dev"]
    explode = [0.1, 0, 0]
    colors = ["#B5EAD7", "#C7CEEA", "#E2F0CB"]

    plt.pie(splits, labels=labels, explode=explode, shadow=True, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.axis("equal")

    # Number of hate or nothate tweets
    not_hate_tweets = df[df["label"] == "nothate"]
    hate_tweets = df[df["label"] == "hate"]
    print(f"Number of hateful sentences in data set: {len(hate_tweets)}")  # 22175
    print(f"Number of not hateful sentences in data set: {len(not_hate_tweets)}\n")  # 18969
    print(not_hate_tweets)

    plt.figure()
    labels = ("Hate", "Not Hate")
    num_of_tweets_types = [len(hate_tweets), len(not_hate_tweets)]
    plt.pie(num_of_tweets_types, colors=["#B5EAD7", "#C7CEEA"], shadow=True, autopct="%1.1f%%", startangle=90)
    plt.legend(labels)
    plt.axis("equal")
    plt.show()


def get_unique_words_vocabulary(dataset: list) -> dict:
    vocabulary = {}
    for tweet in dataset:
        temp_words = tweet.split(" ")
        # temp_words = process_data(tweet, do_stemming=False, do_lemmas=False, do_lowercase=True).split(" ")
        for word in temp_words:
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)

    return vocabulary


def get_max_words_in_sentences(dataset: list) -> int:
    max_len = 0
    max_tweet = ""
    for tweet in dataset:
        temp_tweet = tweet.split(" ")
        # temp_tweet = process_data(tweet, do_stemming=False, do_lemmas=False, do_lowercase=True).split(" ")
        if max_len < len(temp_tweet):
            max_len = len(temp_tweet)
            max_tweet = temp_tweet

    print(f"Max sentence: {max_tweet}")
    return max_len


# print(get_unique_words_vocabulary(train_text))

# Clean: 57001 | No lowercase: 31146 | Lowercase: 25617 | Lowercase & Stemming: 17231 | Lowercase & Lemmas: 22650
print(f"Total number of unique words in dataset: {len(get_unique_words_vocabulary(train_text))}")

# Clean: 408 | No lowercase: 235 | Lowercase: 212 | Lowercase & Stemming: 212 | Lowercase & Lemmas: 212
print(f"Longest sentence in dataset has: {get_max_words_in_sentences(train_text)} words")

plots()
