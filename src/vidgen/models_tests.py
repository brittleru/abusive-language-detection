import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from src.utils import process_data, display_readable_time, display_train_report_and_f1_score, plot_train_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
PRETRAINED_DIR = os.path.join(DATA_DIR, "pretrained")
VIDGEN_DIR = os.path.join(DATA_DIR, "dynamically-hate-vidgen")
DATASET_PATH = os.path.join(VIDGEN_DIR, "Dynamically_Generated_Hate_Dataset_v0.2.3.csv")
MODEL_LOGS_PATH = os.path.join(BASE_DIR, "model-logs")
VIDGEN_MODEL_LOGS_PATH = os.path.join(MODEL_LOGS_PATH, "vidgen")
MODEL_PATH = os.path.join(BASE_DIR, "models")
VIDGEN_MODEL_PATH = os.path.join(MODEL_PATH, "vidgen")

# Clean: 57001 | No lowercase: 31146 | Lowercase: 25617 | Lowercase & Stemming: 17231 | Lowercase & Lemmas: 22650
VOCAB_SIZE = 22650

# Clean: 408 | No lowercase: 235 | Lowercase: 212 | Lowercase & Stemming: 212 | Lowercase & Lemmas: 212
MAX_PADDING_LENGTH = 212


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

    return accuracies, f1_scores


if __name__ == "__main__":

    df = pd.read_csv(DATASET_PATH, delimiter=",")

    train_text = df["text"].tolist()
    train_labels = df["label"].tolist()

    # Transform labels to numerical value
    for index, train_label in enumerate(train_labels):
        if train_label == "hate":
            train_labels[index] = 1
        else:
            train_labels[index] = 0

    for i, text in enumerate(train_text):
        train_text[i] = process_data(text, do_stemming=False, do_lemmas=True, do_lowercase=True)

    tfidfVectorizer = TfidfVectorizer(max_features=1000)
    X = tfidfVectorizer.fit_transform(train_text).toarray()

    X = np.array(X)
    train_labels = np.array(train_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, train_labels, test_size=0.2, random_state=42)

    print("Started training...")

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Gaussian Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(),
    }

    start_time = time.time()
    acc_all, f1_all = classifiers_scores(
        train_set=X_train, train_label=y_train,
        test_set=X_test, test_label=y_test,
        classifier_dict=classifiers
    )
    end_time = time.time()
    display_readable_time(start_time=start_time, end_time=end_time)

    visualization_all(classifier_dict=classifiers, train_result=acc_all, train_result_type="Accuracy")
    visualization_all(classifier_dict=classifiers, train_result=f1_all, train_result_type="F1 Score")
