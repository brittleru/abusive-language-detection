import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from typing import Union
from pathlib import Path
from keras.models import load_model
from keras.metrics import Precision, Recall
from keras.callbacks import CSVLogger, EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFAutoModel, PreTrainedTokenizerFast, \
    PreTrainedTokenizerBase

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

MODEL_FILE_NAME = "hatexplain_large_founta"
# Hate-speech-CNERG/bert-base-uncased-hatexplain | TehranNLP-org/bert-large-hateXplain
HATEXPLAIN_TYPE = "Hate-speech-CNERG/bert-base-uncased-hatexplain"

# Clean: 48 | No lowercase: 36 | Lowercase: 30 | Lowercase & Stemming: 30 | Lowercase & Lemmas: 30
MAX_PADDING_LENGTH = 30
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 10


def encode_tweet(tweet: str, hatexplain_tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizerBase]):
    return hatexplain_tokenizer.encode_plus(
        # tweet,
        process_data(tweet, do_stemming=False, do_lemmas=False, do_lowercase=True),
        add_special_tokens=True,
        max_length=MAX_PADDING_LENGTH,
        truncation=True,
        padding="max_length",
        return_attention_mask=True
    )


def encode_tweets(tweets_text: list, tweets_labels, hatexplain_tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizerBase]):
    if tweets_labels is not None:
        assert len(tweets_text) == len(tweets_labels), f"Features and labels must have the same lengths. " \
                                                       f"Your input ({len(tweets_text)}, {len(tweets_labels)})"

    input_ids = []
    token_type_ids = []
    attention_masks = []

    for tweet in tweets_text:
        tweet_for_hatexplain = encode_tweet(tweet, hatexplain_tokenizer)
        input_ids.append(tweet_for_hatexplain["input_ids"])
        token_type_ids.append(tweet_for_hatexplain["token_type_ids"])
        attention_masks.append(tweet_for_hatexplain["attention_mask"])

    if tweets_labels is not None:
        assert len(input_ids) == len(token_type_ids) == len(attention_masks) == len(tweets_labels), \
            "Arrays must have the same length."
        return np.array(input_ids), np.array(token_type_ids), np.array(attention_masks), np.array(tweets_labels)

    return np.array(input_ids), np.array(token_type_ids), np.array(attention_masks)


def generate_hatexplain_dict(input_ids, token_type_ids, attention_mask) -> dict:
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask
    }


def hatexplain_tuning(hatexplain_type: str = HATEXPLAIN_TYPE):
    input_ids = tf.keras.Input(shape=(MAX_PADDING_LENGTH,), name="input_ids", dtype="int32")
    token_type_ids = tf.keras.Input(shape=(MAX_PADDING_LENGTH,), name="token_type_ids", dtype="int32")
    attention_masks = tf.keras.Input(shape=(MAX_PADDING_LENGTH,), name="attention_mask", dtype="int32")

    hatexplain_model = TFAutoModel.from_pretrained(hatexplain_type)
    encodings = hatexplain_model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)[0]
    last_encoding = tf.squeeze(encodings[:, -1:, :], axis=1)
    # last_encoding = tf.keras.layers.Dropout(0.1)(last_encoding)

    outputs = tf.keras.layers.Dense(4, activation="softmax", name="outputs")(last_encoding)

    temp_model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_masks], outputs=[outputs])
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
    tokenizer = AutoTokenizer.from_pretrained(HATEXPLAIN_TYPE, do_lower_case=True)

    train_df = pd.read_csv(TRAIN_SET_PATH, sep="\t", header=0)
    val_df = pd.read_csv(VAL_SET_PATH, sep="\t", header=0)
    test_df = pd.read_csv(TEST_SET_PATH, sep="\t", header=0)

    train_texts = train_df["sentence"].tolist()
    val_texts = val_df["sentence"].tolist()
    test_texts = test_df["sentence"].tolist()

    train_labels = convert_labels_to_numerical(train_df["class"].tolist())
    val_labels = convert_labels_to_numerical(val_df["class"].tolist())
    test_labels = convert_labels_to_numerical(test_df["class"].tolist())

    train_labels = np_utils.to_categorical(train_labels)
    val_labels = np_utils.to_categorical(val_labels)
    test_labels = np_utils.to_categorical(test_labels)

    train_ids, train_tokens, train_masks, train_labels = encode_tweets(
        tweets_text=train_texts,
        tweets_labels=train_labels,
        hatexplain_tokenizer=tokenizer
    )
    validation_ids, validation_tokens, validation_masks, val_labels = encode_tweets(
        tweets_text=val_texts,
        tweets_labels=val_labels,
        hatexplain_tokenizer=tokenizer
    )
    test_ids, test_tokens, test_masks, test_labels = encode_tweets(
        tweets_text=test_texts,
        tweets_labels=test_labels,
        hatexplain_tokenizer=tokenizer
    )

    train_data = generate_hatexplain_dict(train_ids, train_tokens, train_masks)
    validation_data = (generate_hatexplain_dict(validation_ids, validation_tokens, validation_masks), val_labels)
    test_data = generate_hatexplain_dict(test_ids, test_tokens, test_masks)

    model = hatexplain_tuning()
    print(model.summary())
    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=2, restore_best_weights=True)
    csv_logger = CSVLogger(os.path.join(FOUNTA_MODEL_LOGS_PATH, f"{MODEL_FILE_NAME}.log"), separator=",",
                           append=False)
    start_time = time.time()
    hist = model.fit(train_data, train_labels, validation_data=validation_data, epochs=EPOCHS, batch_size=BATCH_SIZE,
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
    # new_model = load_model(os.path.join(FOUNTA_MODEL_PATH, f"{MODEL_FILE_NAME}.h5"))

    # predictions = new_model.predict(test_texts)
    predictions = model.predict(test_data)

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
