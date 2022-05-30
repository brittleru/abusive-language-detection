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
PRETRAINED_DIR = os.path.join(DATA_DIR, "pretrained")
DAVIDSON_DIR = os.path.join(DATA_DIR, "hatespeech-davidson")
DATASET_PATH = os.path.join(DAVIDSON_DIR, "data/labeled_data.csv")
MODEL_LOGS_PATH = os.path.join(BASE_DIR, "model-logs")
DAVIDSON_MODEL_LOGS_PATH = os.path.join(MODEL_LOGS_PATH, "davidson")
MODEL_PATH = os.path.join(BASE_DIR, "models")
DAVIDSON_MODEL_PATH = os.path.join(MODEL_PATH, "davidson")


# Clean: 59462 | No lowercase: 24667 | Lowercase: 19586 | Lowercase & Stemming: 15283 | Lowercase & Lemmas: 17819
VOCAB_SIZE = 17819

# Clean: 36 | No lowercase: 29 | Lowercase: 28 | Lowercase & Stemming: 28 | Lowercase & Lemmas: 28
MAX_PADDING_LENGTH = 28


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
        f1 = f1_score(test_label, train_predictions, average='weighted')
        accuracies.append(acc)
        f1_scores.append(f1)
        print(f"{key} -> acc: {acc} | f1: {f1}")
        print(f"Classification Report for {key}")
        print(f"{classification_report(test_label, train_predictions)}\n")

    return accuracies, f1_scores


if __name__ == "__main__":

    df = pd.read_csv(DATASET_PATH, delimiter=",")

    train_text = df["tweet"].tolist()
    train_labels = df["class"].tolist()

    for i, text in enumerate(train_text):
        train_text[i] = process_data(text, do_stemming=False, do_lemmas=True, do_lowercase=True)

    tfidfVectorizer = TfidfVectorizer(max_features=1000)
    X = tfidfVectorizer.fit_transform(train_text).toarray()

    X = np.array(X)
    train_labels = np.array(train_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, train_labels, test_size=0.2, random_state=42)

    print("Started training...")

    classifiers = {
        "One vs All with Logistic Regression": OneVsRestClassifier(LogisticRegression(random_state=0, solver='liblinear')),
        "One vs One with 3 KNN": OneVsOneClassifier(KNeighborsClassifier(n_neighbors=3)),
        "XGBoost": XGBClassifier(),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(silent=True),
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
