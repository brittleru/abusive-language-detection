import os
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from keras.metrics import Precision, Recall
from keras.preprocessing.text import one_hot
from keras.models import Sequential, load_model
from sklearn.metrics import classification_report
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Conv1D, Dropout, SpatialDropout1D, MaxPooling1D, Flatten

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

# cnn_model_no_preprocess | cnn_model_no_lowercase | cnn_model_lowercase |
# cnn_model_lowercase_stemming | cnn_model_lowercase_lemme
MODEL_FILE_NAME = "cnn_model_lowercase_lemme"

# Clean: 42731 | No lowercase: 25045 | Lowercase: 19449 | Lowercase & Stemming: 14300 | Lowercase & Lemmas: 17613
VOCAB_SIZE = 17613

# Clean: 103 | No lowercase: 60 | Lowercase: 42 | Lowercase & Stemming: 42 | Lowercase & Lemmas: 42
MAX_PADDING_LENGTH = 42

LEARNING_RATE = 2e-5  # 0.0001 | 2e-5
EPOCHS = 20
BATCH_SIZE = 32
HYPER_PARAMETERS = {
    "filters": [32, 64, 128, 254],
    "kernel_size": [3, 5, 7, 9]
}


def encode_one_hot_and_preprocess(texts: list, vocab_size: int = VOCAB_SIZE) -> list:
    return [one_hot(
        # temp_text,
        process_data(temp_text, do_stemming=False, do_lemmas=True, do_lowercase=True),
        vocab_size,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=False,
        split=' '
    ) for temp_text in texts]


def prepare_data_for_train(texts: list, max_len: int = MAX_PADDING_LENGTH) -> np.ndarray:
    texts = encode_one_hot_and_preprocess(texts)
    texts = tf.keras.preprocessing.sequence.pad_sequences(texts, maxlen=max_len, padding="post")

    return np.array(texts)


def cnn_tuning(filters, kernel_size):
    # # OVER-FIT
    # temp_model = Sequential([
    #     Embedding(VOCAB_SIZE, 8, input_length=MAX_PADDING_LENGTH),
    #     Conv1D(filters, kernel_size, activation="relu"),
    #     GlobalMaxPool1D(),
    #     # Dense(128, activation="relu"),
    #     # Dropout(0.5),
    #     # Dense(64, activation="relu"),
    #     # Dropout(0.1),
    #     Dense(1, activation="sigmoid")
    # ])

    temp_model = Sequential([
        Embedding(VOCAB_SIZE, 150, input_length=MAX_PADDING_LENGTH),

        Conv1D(128, kernel_size=5, padding='same', activation="relu"),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=5, padding='same', activation="relu"),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=5, padding='same', activation="relu"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation="relu"),
        # Dropout(0.5),
        # Dense(10, activation="relu"),
        # Dropout(0.1),
        Dense(1, activation="sigmoid")
    ])
    temp_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall()]
    )
    return temp_model


def convert_labels_to_numerical(labels: list):
    # Transform labels to numerical value
    for index, label in enumerate(labels):
        if label == "NOT":
            labels[index] = 0
        elif label == "OFF":
            labels[index] = 1
        else:
            raise ValueError("Class column must have only 'NOT' or 'OFF' values")

    return labels


if __name__ == "__main__":

    train_df = pd.read_csv(TRAIN_SET_PATH, delimiter=",")
    # test_df = pd.read_csv(TEST_SET_PATH, delimiter=",")

    train_texts = train_df["tweet"].tolist()
    # test_texts = test_df["tweet"].tolist()

    train_labels = convert_labels_to_numerical(train_df["subtask_a"].tolist())
    # test_labels = convert_labels_to_numerical(test_df["subtask_a"].tolist())

    train_texts = prepare_data_for_train(train_texts)
    # test_texts = prepare_data_for_train(test_texts)

    train_labels = np.array(train_labels)
    # test_labels = np.array(test_labels)

    X_train, X_temp, y_train, y_temp = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # print("\n\n")
    # print(X_train.shape, y_train.shape)
    # print(X_val.shape, y_val.shape)
    # print(X_test.shape, y_test.shape)

    val_data = (X_val, y_val)
    model = cnn_tuning(64, 9)
    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5, restore_best_weights=True)
    csv_logger = CSVLogger(os.path.join(ZAMPIERI_MODEL_LOGS_PATH, f"{MODEL_FILE_NAME}.log"), separator=",",
                           append=False)
    start_time = time.time()
    hist = model.fit(X_train, y_train, validation_data=val_data, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     callbacks=[csv_logger, early_stop])
    end_time = time.time()
    model.save(os.path.join(ZAMPIERI_MODEL_PATH, f"{MODEL_FILE_NAME}.h5"))
    display_readable_time(start_time=start_time, end_time=end_time)

    log_data = pd.read_csv(os.path.join(ZAMPIERI_MODEL_LOGS_PATH, f"{MODEL_FILE_NAME}.log"), sep=",", engine="python")
    display_train_report_and_f1_score(log_data)
    plot_train_data(log_data, train_metric="accuracy", validation_metric="val_accuracy")
    plot_train_data(log_data, train_metric="loss", validation_metric="val_loss")
    plot_train_data(log_data, train_metric="precision", validation_metric="val_precision")
    plot_train_data(log_data, train_metric="recall", validation_metric="val_recall")
    # plt.show()

    # # ======= Test Model =======
    # new_model = load_model(os.path.join(ZAMPIERI_MODEL_PATH, f"{MODEL_FILE_NAME}.h5"))
    # predictions = new_model.predict(X_test)
    predictions = model.predict(X_test)

    for prediction in predictions:
        for index, pred_class in enumerate(prediction):
            if pred_class > 0.5:
                prediction[index] = 1
            else:
                prediction[index] = 0

    # print(predictions)
    # print(test_labels)
    print(type(y_test), type(predictions))
    # print(predictions)
    # probVal = np.amax(predictions)
    # classIndex = np.argmax(predictions, axis=1)[0]
    print(f"\n{classification_report(y_test, predictions)}")

    # # ======= Grid Search =======
    # model = tf.keras.wrappers.scikit_learn.KerasClassifier(
    #     build_fn=cnn_tuning,
    #     epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    # )
    # start_time = time.time()
    # search = GridSearchCV(estimator=model, param_grid=HYPER_PARAMETERS, cv=5, verbose=1)
    # search_result = search.fit(X_train, y_train)
    # test_accuracy = search.score(X_temp, y_temp)
    # end_time = time.time()
    # print("\n\n")
    # display_readable_time(start_time=start_time, end_time=end_time)
    # print(search.best_params_)
    # print(search.best_estimator_)
    # print(search.best_index_)
    # print(search.best_score_)
    # print(test_accuracy)
    #
    # joblib.dump(search.best_estimator_, os.path.join(MODEL_PATH, "grid_model_cnn.pkl"))
    # try:
    #     search.save(os.path.join(MODEL_PATH, "grid_model_cnn.h5"))
    # except:
    #     print("Something went wrong")
    # load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    # best_model = joblib.load(os.path.join(MODEL_PATH, "grid_model_cnn.pkl"))
