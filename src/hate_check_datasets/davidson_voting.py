import os
import gc
import time
import joblib
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from keras.metrics import Precision, Recall
from keras.preprocessing.text import one_hot
from keras.models import Sequential, load_model
from keras.utils import np_utils
from sklearn.metrics import classification_report

from src.utils import process_data, display_readable_time, display_train_report_and_f1_score, plot_train_data
from src.davidson import cnn_one_hot_davidson, bert_davidson, xlnet_davidson, bertweet_davidson, \
    roberta_davidson, hatexplain_davidson, hatebert_davidson

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
HATE_CHECK_DIR = os.path.join(DATA_DIR, "hatecheck-data")
DATASET_PATH = os.path.join(HATE_CHECK_DIR, "test_suite_cases.csv")
MODEL_LOGS_PATH = os.path.join(BASE_DIR, "model-logs")
MODEL_PATH = os.path.join(BASE_DIR, "models")

DAVIDSON_MODEL_PATH = os.path.join(MODEL_PATH, "davidson")
DAVIDSON_CNN_MODEL_PATH = os.path.join(DAVIDSON_MODEL_PATH, "cnn_model_lowercase_lemme.h5")


def convert_hatecheck_to_numerical(labels: list) -> list:
    # Transform labels to numerical value
    for index, label in enumerate(labels):
        if label == "non-hateful":
            labels[index] = 0
        elif label == "hateful":
            labels[index] = 1
        else:
            raise ValueError("Class column must have only 'non-hateful', or 'hateful' values")

    return labels


def compute_hatecheck_predictions(model_preds, final_preds):
    for prediction, final_pred in zip(model_preds, final_preds):
        for index, pred_class in enumerate(prediction):
            if pred_class == max(prediction):
                prediction[index] = 1
            else:
                prediction[index] = 0

            if index == 0:
                if prediction[0] == 1.0:
                    final_pred[1] = 1.0
                else:
                    if prediction[1] == 1.0:
                        final_pred[1] = 1.0
                    else:
                        final_pred[1] = 0.0
            elif index == 1:
                if prediction[1] == 1.0:
                    final_pred[1] = 1.0
                else:
                    if prediction[0] == 1.0:
                        final_pred[1] = 1.0
                    else:
                        final_pred[1] = 0.0
            elif index == 2:
                if prediction[2] == 1.0:
                    final_pred[0] = 1.0
                else:
                    final_pred[0] = 0.0

    return model_preds, final_preds


if __name__ == "__main__":

    # Clean: 1828 | No lowercase: 1316 | Lowercase: 1179 | Lowercase & Stemming: 1004 | Lowercase & Lemmas: 1114
    hate_check_vocab = 1114
    # Clean: 20 | No lowercase: 12 | Lowercase: 11 | Lowercase & Stemming: 11 | Lowercase & Lemmas: 11
    hate_check_padding = 11
    df = pd.read_csv(DATASET_PATH, delimiter=",")

    hatecheck_texts = df["test_case"].tolist()
    hatecheck_labels = convert_hatecheck_to_numerical(df["label_gold"].tolist())

    cnn_hatecheck_labels = np_utils.to_categorical(hatecheck_labels)

    cnn_hatecheck_texts = cnn_one_hot_davidson.prepare_data_for_train(hatecheck_texts, vocab_size=hate_check_vocab)
    cnn_hatecheck_labels = np.array(cnn_hatecheck_labels)

    davidson_cnn_model = load_model(DAVIDSON_CNN_MODEL_PATH)

    cnn_predictions = davidson_cnn_model.predict(cnn_hatecheck_texts)
    del davidson_cnn_model
    gc.collect()

    cnn_predictions_for_hatecheck = np.ndarray((cnn_predictions.shape[0], 2))

    mod_preds, cnn_preds = compute_hatecheck_predictions(cnn_predictions, cnn_predictions_for_hatecheck)



    print(cnn_preds)
    print(cnn_preds)

    print()
    print(cnn_hatecheck_labels)
    print(type(cnn_hatecheck_labels), type(cnn_preds))
    # print(predictions)
    # probVal = np.amax(predictions)
    # classIndex = np.argmax(predictions, axis=1)[0]
    print(f"\n{classification_report(cnn_hatecheck_labels, cnn_preds)}")
