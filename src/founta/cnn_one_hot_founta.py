import os
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from keras.utils import np_utils
from keras.metrics import Precision, Recall
from keras.models import Sequential, load_model
from sklearn.metrics import classification_report
from keras.callbacks import CSVLogger, EarlyStopping
from keras.preprocessing.text import one_hot, Tokenizer
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Conv1D, Dropout, SpatialDropout1D, MaxPooling1D, Flatten

from src.utils import process_data, display_readable_time, display_train_report_and_f1_score, plot_train_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
PRETRAINED_DIR = os.path.join(DATA_DIR, "pretrained")
FOUNTA_DIR = os.path.join(DATA_DIR, "large-founta")
TRAIN_SET_PATH = os.path.join(FOUNTA_DIR, "train.tsv")
VAL_SET_PATH = os.path.join(FOUNTA_DIR, "dev.tsv")
TEST_SET_PATH = os.path.join(FOUNTA_DIR, "test.tsv")
MODEL_LOGS_PATH = os.path.join(BASE_DIR, "model-logs")
FOUNTA_MODEL_LOGS_PATH = os.path.join(MODEL_LOGS_PATH, "founta")
MODEL_PATH = os.path.join(BASE_DIR, "models")
FOUNTA_MODEL_PATH = os.path.join(MODEL_PATH, "founta")
GLOVE_PATH = os.path.join(PRETRAINED_DIR, "glove-6B")
GLOVE_PATH_6B300D = os.path.join(GLOVE_PATH, "glove.6B.50d.txt")

# cnn_model_no_preprocess | cnn_model_no_lowercase | cnn_model_lowercase |
# cnn_model_lowercase_stemming | cnn_model_lowercase_lemme
MODEL_FILE_NAME = "cnn_glove_model_lowercase_lemme"

# Clean: 292512 | No lowercase: 97205 | Lowercase: 71500 | Lowercase & Stemming: 56824 | Lowercase & Lemmas: 66265
VOCAB_SIZE = 66265

# Clean: 48 | No lowercase: 36 | Lowercase: 30 | Lowercase & Stemming: 30 | Lowercase & Lemmas: 30
MAX_PADDING_LENGTH = 50
OOV_TOKEN = "<OOV>"
TOKENIZER = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)

LEARNING_RATE = 2e-5  # 0.0001 | 2e-5
EPOCHS = 30
BATCH_SIZE = 32
HYPER_PARAMETERS = {
    "filters": [32, 64, 128, 254],
    "kernel_size": [3, 5, 7, 9]
}


def create_embeddings(path: str = GLOVE_PATH_6B300D) -> dict:
    embeddings_index = {}

    f = open(path, "r", errors='ignore', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefficients
    f.close()

    return embeddings_index


def create_embedding_matrix(word_index: dict, max_len: int = MAX_PADDING_LENGTH) -> np.ndarray:
    embeddings_index = create_embeddings()
    embedding_matrix = np.zeros((len(word_index) + 1, max_len))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def prepare_data_for_train(texts: list, max_len: int = MAX_PADDING_LENGTH) -> tuple:
    for i, text in enumerate(texts):
        # texts[i] = text
        # texts[i] = process_data(text, do_stemming=False, do_lemmas=False, do_lowercase=False)
        texts[i] = process_data(text, do_stemming=False, do_lemmas=True, do_lowercase=True)

    TOKENIZER.fit_on_texts(texts)
    WORD_INDEX = TOKENIZER.word_index
    texts_sequences = TOKENIZER.texts_to_sequences(texts)
    texts_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        texts_sequences, maxlen=max_len, padding="post", truncating="post"
    )

    return np.array(texts_sequences), WORD_INDEX


def cnn_tuning(filters: int, kernel_size: int, embedding_matrix: np.ndarray, word_index: dict):
    # temp_model = Sequential([
    #     Embedding(
    #         input_dim=len(word_index) + 1,
    #         output_dim=MAX_PADDING_LENGTH,
    #         weights=[embedding_matrix],
    #         input_length=MAX_PADDING_LENGTH,
    #         trainable=False
    #     ),
    #     Conv1D(filters, kernel_size, activation="relu"),
    #     GlobalMaxPool1D(),
    #     Dense(128, activation="relu"),
    #     Dropout(0.5),
    #     Dense(64, activation="relu"),
    #     Dropout(0.1),
    #     Dense(4, activation="softmax")
    # ])
    temp_model = Sequential([
        Embedding(
            input_dim=len(word_index) + 1,
            output_dim=MAX_PADDING_LENGTH,
            weights=[embedding_matrix],
            input_length=MAX_PADDING_LENGTH,
            trainable=False
        ),
        Conv1D(128, kernel_size=9, padding='same', activation="relu"),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=9, padding='same', activation="relu"),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=9, padding='same', activation="relu"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation="relu"),
        # Dropout(0.5),
        # Dense(10, activation="relu"),
        # Dropout(0.1),
        Dense(4, activation="softmax")
    ])
    temp_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", Precision(), Recall()]
    )
    return temp_model


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


if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_SET_PATH, sep="\t", header=0)
    val_df = pd.read_csv(VAL_SET_PATH, sep="\t", header=0)
    test_df = pd.read_csv(TEST_SET_PATH, sep="\t", header=0)

    # If you prefer your own distribution of TRAIN / VAL / TEST
    dataset_df = pd.concat([train_df, val_df, test_df])

    train_texts = train_df["sentence"].tolist()
    val_texts = val_df["sentence"].tolist()
    test_texts = test_df["sentence"].tolist()

    train_labels = convert_labels_to_numerical(train_df["class"].tolist())
    val_labels = convert_labels_to_numerical(val_df["class"].tolist())
    test_labels = convert_labels_to_numerical(test_df["class"].tolist())

    train_labels = np_utils.to_categorical(train_labels)
    val_labels = np_utils.to_categorical(val_labels)
    test_labels = np_utils.to_categorical(test_labels)

    train_texts, train_word_idx = prepare_data_for_train(train_texts)
    val_texts, val_word_idx = prepare_data_for_train(val_texts)
    test_texts, test_word_idx = prepare_data_for_train(test_texts)

    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    print(f"\nTrain shapes: {train_texts.shape} | {train_labels.shape}")
    print(f"Validation shapes: {val_texts.shape} | {val_labels.shape}")
    print(f"Test shapes: {test_texts.shape} | {test_labels.shape}\n")

    embd_matrix = create_embedding_matrix(word_index=train_word_idx)


    val_data = (val_texts, val_labels)

    model = cnn_tuning(64, 9, embedding_matrix=embd_matrix, word_index=train_word_idx)
    csv_logger = CSVLogger(os.path.join(FOUNTA_MODEL_LOGS_PATH, f"{MODEL_FILE_NAME}.log"), separator=",",
                           append=False)
    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=2, restore_best_weights=True)
    start_time = time.time()
    hist = model.fit(train_texts, train_labels, validation_data=val_data, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     callbacks=[csv_logger, early_stop])
    end_time = time.time()
    model.save(os.path.join(FOUNTA_MODEL_PATH, f"{MODEL_FILE_NAME}.h5"))
    display_readable_time(start_time=start_time, end_time=end_time)

    log_data = pd.read_csv(os.path.join(FOUNTA_MODEL_LOGS_PATH, f"{MODEL_FILE_NAME}.log"), sep=",", engine="python")
    display_train_report_and_f1_score(log_data)
    plot_train_data(log_data, train_metric="accuracy", validation_metric="val_accuracy")
    plot_train_data(log_data, train_metric="loss", validation_metric="val_loss")
    plot_train_data(log_data, train_metric="precision", validation_metric="val_precision")
    plot_train_data(log_data, train_metric="recall", validation_metric="val_recall")
    plt.show()

    # # ======= Test Model =======
    new_model = load_model(os.path.join(FOUNTA_MODEL_PATH, f"{MODEL_FILE_NAME}.h5"))
    predictions = new_model.predict(test_texts)

    # predictions = model.predict(test_texts)

    for prediction in predictions:
        for index, pred_class in enumerate(prediction):
            if pred_class == max(prediction):
                prediction[index] = 1
            else:
                prediction[index] = 0

    print(predictions)
    print(test_labels)
    print(len(predictions), len(test_labels))
    print(type(test_labels), type(predictions))

    print(f"\n{classification_report(test_labels, predictions)}")

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









