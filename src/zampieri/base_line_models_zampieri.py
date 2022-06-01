import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from src.utils import process_data, display_readable_time, display_train_report_and_f1_score, plot_train_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
PRETRAINED_DIR = os.path.join(DATA_DIR, "pretrained")
ZAMPIERI_DIR = os.path.join(DATA_DIR, "olid-zampieri")
TRAIN_SET_PATH = os.path.join(ZAMPIERI_DIR, "cleaned_train_data_v1.csv")
TEST_SET_PATH = os.path.join(ZAMPIERI_DIR, "cleaned_test_data_v1.csv")
MODEL_LOGS_PATH = os.path.join(BASE_DIR, "model-logs")
ZAMPIERI_MODEL_LOGS_PATH = os.path.join(MODEL_LOGS_PATH, "zampieri")
MODEL_PATH = os.path.join(BASE_DIR, "models")
ZAMPIERI_MODEL_PATH = os.path.join(MODEL_PATH, "zampieri")

# Clean: 42731 | No lowercase: 25045 | Lowercase: 19449 | Lowercase & Stemming: 14300 | Lowercase & Lemmas: 17613
VOCAB_SIZE = 17613

# Clean: 103 | No lowercase: 60 | Lowercase: 42 | Lowercase & Stemming: 42 | Lowercase & Lemmas: 42
MAX_PADDING_LENGTH = 42


def convert_labels_to_numerical(labels: list):
    # Transform labels to numerical value
    for index, label in enumerate(labels):
        if label == "NOT":
            labels[index] = 0
        elif label == "OFF":
            labels[index] = 1
        else:
            raise ValueError("Class column must have only 'none', 'racism', or 'sexism' values")

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
        classifier = classifier_dict[key].fit(train_set, train_label)
        train_predictions = classifier.predict(test_set)
        acc = accuracy_score(test_label, train_predictions)
        f1 = f1_score(test_label, train_predictions)
        accuracies.append(acc)
        f1_scores.append(f1)
        print(f"{key} -> acc: {acc} | f1: {f1}")
        print(f"Classification Report for {key}")
        print(f"{classification_report(test_label, train_predictions)}\n")

    return accuracies, f1_scores


if __name__ == "__main__":

    train_df = pd.read_csv(TRAIN_SET_PATH, delimiter=",")
    # test_df = pd.read_csv(TEST_SET_PATH, delimiter=",")

    train_texts = train_df["tweet"].tolist()
    # test_texts = test_df["tweet"].tolist()

    train_labels = convert_labels_to_numerical(train_df["subtask_a"].tolist())
    # test_labels = convert_labels_to_numerical(test_df["subtask_a"].tolist())

    for i, text in enumerate(train_texts):
        train_texts[i] = process_data(str(text), do_stemming=False, do_lemmas=False, do_lowercase=False)
        # train_texts[i] = str(text)

    # for i, text in enumerate(test_texts):
    #     test_texts[i] = process_data(str(text), do_stemming=False, do_lemmas=False, do_lowercase=False)
    #     # test_texts[i] = str(text)

    tfidfVectorizer = TfidfVectorizer(max_features=MAX_PADDING_LENGTH)
    train_texts = tfidfVectorizer.fit_transform(train_texts).toarray()
    # test_texts = tfidfVectorizer.fit_transform(test_texts).toarray()

    train_texts = np.array(train_texts)
    train_labels = np.array(train_labels)
    # test_texts = np.array(test_texts)
    # test_labels = np.array(test_labels)

    X_train, test_texts, y_train, test_labels = train_test_split(train_texts, train_labels, test_size=0.1)

    print("Started training...")

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Gaussian Naive Bayes": GaussianNB(),
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
