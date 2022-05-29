import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from src.utils import process_data, display_readable_time, display_train_report_and_f1_score, plot_train_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
FOUNTA_DIR = os.path.join(DATA_DIR, "large-founta")
TRAIN_SET_PATH = os.path.join(FOUNTA_DIR, "train.tsv")
VAL_SET_PATH = os.path.join(FOUNTA_DIR, "dev.tsv")
TEST_SET_PATH = os.path.join(FOUNTA_DIR, "test.tsv")
MODEL_LOGS_PATH = os.path.join(BASE_DIR, "model-logs")
FOUNTA_MODEL_LOGS_PATH = os.path.join(MODEL_LOGS_PATH, "founta")
MODEL_PATH = os.path.join(BASE_DIR, "models")
FOUNTA_MODEL_PATH = os.path.join(MODEL_PATH, "founta")

# Clean: 292512 | No lowercase: 97205 | Lowercase: 71500 | Lowercase & Stemming: 56824 | Lowercase & Lemmas: 66265
VOCAB_SIZE = 66265

# Clean: 48 | No lowercase: 36 | Lowercase: 30 | Lowercase & Stemming: 30 | Lowercase & Lemmas: 30
MAX_PADDING_LENGTH = 30


def convert_labels_to_numerical(labels: list):
    # Transform labels to numerical value
    for index, label in enumerate(labels):
        if label == "normal":
            labels[index] = 0
        elif label == "spam":
            labels[index] = 1
        elif label == "abusive":
            labels[index] = 2
        elif label == "hateful":
            labels[index] = 3
        else:
            raise ValueError("Class column must have only 'normal', 'spam', 'abusive' or 'hateful' values")

    return labels


def visualization_all(classifier_dict: dict, train_result: list, train_result_type: str) -> None:
    plt.xlabel("Classifiers")
    plt.ylabel(train_result_type.capitalize())
    width = 0.6
    plt.bar(list(classifier_dict.keys()), train_result, width, color=(0.2, 0.4, 0.6, 0.6))
    for i, v in enumerate(train_result):
        plt.text(i - .15,
                 v / train_result[i] / 3,
                 round(train_result[i], 2),
                 fontsize=12
                 # color=label_color_list[i]
                 )

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def classifiers_scores(train_set: np.ndarray, train_label: np.ndarray,
                       test_set: np.ndarray, test_label: np.ndarray,
                       classifier_dict: dict) -> tuple:
    accuracies = []
    f1_scores = []

    for key in classifier_dict:
        print(f"Training classifier: {key}")
        classifier = classifier_dict[key].fit(train_set, train_label)
        train_predictions = classifier.predict(test_set)
        acc = accuracy_score(test_label, train_predictions)
        f1 = f1_score(test_label, train_predictions, average='weighted')
        accuracies.append(acc)
        f1_scores.append(f1)
        print(f"{key} -> acc: {acc} | f1: {f1}")
        print(f"Classification Report for {key}")
        print(f"{classification_report(test_label, train_predictions)}\n")

    return accuracies, f1_scores


if __name__ == "__main__":

    train_df = pd.read_csv(TRAIN_SET_PATH, sep="\t", header=0)
    val_df = pd.read_csv(VAL_SET_PATH, sep="\t", header=0)
    test_df = pd.read_csv(TEST_SET_PATH, sep="\t", header=0)

    dataset_df = pd.concat([train_df, val_df])
    dataset_texts = dataset_df["sentence"].tolist()
    test_texts = test_df["sentence"].tolist()

    dataset_labels = dataset_df["class"].tolist()
    test_labels = test_df["class"].tolist()

    dataset_labels = convert_labels_to_numerical(dataset_labels)
    test_labels = convert_labels_to_numerical(test_labels)

    for i, text in enumerate(dataset_texts):
        dataset_texts[i] = process_data(text, do_stemming=False, do_lemmas=True, do_lowercase=True)

    for i, text in enumerate(test_texts):
        test_texts[i] = process_data(text, do_stemming=False, do_lemmas=True, do_lowercase=True)

    tfidfVectorizer = TfidfVectorizer(max_features=MAX_PADDING_LENGTH)
    train_texts = tfidfVectorizer.fit_transform(dataset_texts).toarray()
    test_texts = tfidfVectorizer.fit_transform(test_texts).toarray()

    train_texts = np.array(train_texts)
    test_texts = np.array(test_texts)
    train_labels = np.array(dataset_labels)
    test_labels = np.array(test_labels)

    print(train_texts.shape, train_labels.shape)
    print(test_texts.shape, test_labels.shape)

    print("\n")
    for i in range(10):
        print(train_texts[i])

    for i in range(10):
        print(train_labels[i])

    print("Started training...")

    classifiers = {
        "One vs All with Logistic Regression": OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')),
        "One vs One with 4 KNN": OneVsOneClassifier(KNeighborsClassifier(n_neighbors=4)),
        "XGBoost": XGBClassifier(),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(silent=True),
    }

    start_time = time.time()
    acc_all, f1_all = classifiers_scores(
        train_set=train_texts, train_label=train_labels,
        test_set=test_texts, test_label=test_labels,
        classifier_dict=classifiers
    )
    end_time = time.time()
    display_readable_time(start_time=start_time, end_time=end_time)

    visualization_all(classifier_dict=classifiers, train_result=acc_all, train_result_type="Accuracy")
    visualization_all(classifier_dict=classifiers, train_result=f1_all, train_result_type="F1 Score")
