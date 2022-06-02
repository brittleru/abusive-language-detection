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
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Conv1D, Dropout, SpatialDropout1D, MaxPooling1D, Flatten

from src.utils import process_data, display_readable_time, display_train_report_and_f1_score, plot_train_data
from src.wassem import cnn_one_hot_wassem, bert_wassem, xlnet_wassem, bertweet_wassem, \
    roberta_wassem, hatexplain_wassem, hatebert_wassem

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


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH, delimiter=",")

    hatecheck_texts = df["test_case"].tolist()
    hatecheck_labels = convert_hatecheck_to_numerical(df["label_gold"].tolist())

    cnn_hatecheck_labels = np_utils.to_categorical(hatecheck_labels)

    cnn_hatecheck_texts = cnn_one_hot_davidson.prepare_data_for_train(hatecheck_texts)
    cnn_hatecheck_labels = np.array(cnn_hatecheck_labels)

    davidson_cnn_model = load_model(DAVIDSON_CNN_MODEL_PATH)

    cnn_predictions = davidson_cnn_model.predict(cnn_hatecheck_texts)
    del davidson_cnn_model
    gc.collect()

    cnn_predictions_for_hatecheck = np.ndarray((cnn_predictions.shape[0], 2))

    for prediction, final_pred in zip(cnn_predictions, cnn_predictions_for_hatecheck):
        for index, pred_class in enumerate(prediction):
            if pred_class == max(prediction):
                prediction[index] = 1
            else:
                prediction[index] = 0

            if index == 0:
                if prediction[index] == 1.0:
                    final_pred[1] = 1.0
                else:
                    final_pred[1] = 0.0
            elif index == 1:
                if prediction[index] == 1.0:
                    final_pred[1] = 1.0
                else:
                    final_pred[1] = 0.0
            elif index == 2:
                if prediction[index] == 1.0:
                    final_pred[0] = 1.0
                else:
                    final_pred[0] = 0.0



    print(cnn_predictions)
    print(cnn_predictions_for_hatecheck)

    print()
    print(cnn_hatecheck_labels)
    print(type(cnn_hatecheck_labels), type(cnn_predictions))
    # print(predictions)
    # probVal = np.amax(predictions)
    # classIndex = np.argmax(predictions, axis=1)[0]
    print(f"\n{classification_report(cnn_hatecheck_labels, cnn_predictions_for_hatecheck)}")
