import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union
from keras.utils import np_utils
from keras.models import load_model
from keras.metrics import Precision, Recall
from sklearn.metrics import classification_report
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase, RobertaTokenizer, TFRobertaModel

from src.utils import process_data, display_readable_time, display_train_report_and_f1_score, plot_train_data


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDGEN_DIR = os.path.join(DATA_DIR, "dynamically-hate-vidgen")
DATASET_PATH = os.path.join(VIDGEN_DIR, "Dynamically_Generated_Hate_Dataset_v0.2.3.csv")
MODEL_LOGS_PATH = os.path.join(BASE_DIR, "model-logs")
VIDGEN_MODEL_LOGS_PATH = os.path.join(MODEL_LOGS_PATH, "vidgen")
MODEL_PATH = os.path.join(BASE_DIR, "models")
VIDGEN_MODEL_PATH = os.path.join(MODEL_PATH, "vidgen")

MODEL_FILE_NAME = "roberta_large_vidgen"
ROBERTA_TYPE = "roberta-large"  # roberta-base | roberta-large | roberta-large-mnli

# Clean: 57001 | No lowercase: 31146 | Lowercase: 25617 | Lowercase & Stemming: 17231 | Lowercase & Lemmas: 22650
VOCAB_SIZE = 22650
# Clean: 408 | No lowercase: 235 | Lowercase: 212 | Lowercase & Stemming: 212 | Lowercase & Lemmas: 212
MAX_PADDING_LENGTH = 212

LEARNING_RATE = 2e-5  # 0.0001 | 2e-5
BATCH_SIZE = 32
EPOCHS = 10


def encode_tweet(tweet: str, roberta_tokenizer: PreTrainedTokenizerBase):
    return roberta_tokenizer.encode_plus(
        # tweet,
        process_data(tweet, do_stemming=False, do_lemmas=False, do_lowercase=True),
        add_special_tokens=True,
        max_length=MAX_PADDING_LENGTH,
        truncation=True,
        padding="max_length",
        return_attention_mask=True
    )


def encode_tweets(tweets_text: list, tweets_labels, roberta_tokenizer: PreTrainedTokenizerBase):
    if tweets_labels is not None:
        assert len(tweets_text) == len(tweets_labels), f"Features and labels must have the same lengths. " \
                                                       f"Your input ({len(tweets_text)}, {len(tweets_labels)})"

    input_ids = []
    attention_masks = []

    for tweet in tweets_text:
        tweet_for_roberta = encode_tweet(tweet, roberta_tokenizer)
        input_ids.append(tweet_for_roberta["input_ids"])
        attention_masks.append(tweet_for_roberta["attention_mask"])

    if tweets_labels is not None:
        assert len(input_ids) == len(attention_masks) == len(tweets_labels), \
            "Arrays must have the same length."
        return np.array(input_ids), np.array(attention_masks), np.array(tweets_labels)

    return np.array(input_ids), np.array(attention_masks)


def generate_roberta_dict(input_ids, attention_mask) -> dict:
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def roberta_tuning(roberta_type: str = ROBERTA_TYPE):
    input_ids = tf.keras.Input(shape=(MAX_PADDING_LENGTH,), name="input_ids", dtype="int32")
    attention_masks = tf.keras.Input(shape=(MAX_PADDING_LENGTH,), name="attention_mask", dtype="int32")

    roberta_model = TFRobertaModel.from_pretrained(roberta_type)
    encodings = roberta_model(input_ids=input_ids, attention_mask=attention_masks)[0]
    last_encoding = tf.squeeze(encodings[:, -1:, :], axis=1)
    # last_encoding = tf.keras.layers.Dropout(0.1)(last_encoding)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="outputs")(last_encoding)

    temp_model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=outputs)
    temp_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall()]
    )

    return temp_model


def convert_labels_to_numerical(labels: list):
    # Transform labels to numerical value
    for index, label in enumerate(labels):
        if label == "nothate":
            labels[index] = 0
        elif label == "hate":
            labels[index] = 1
        else:
            raise ValueError("Class column must have only 'nothate' or 'hate' values")

    return labels


if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_TYPE, do_lower_case=True)

    df = pd.read_csv(DATASET_PATH, delimiter=",")

    train_text = df["text"].tolist()
    train_labels = convert_labels_to_numerical(df["label"].tolist())

    X_train, X_temp, y_train, y_temp = train_test_split(train_text, train_labels, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    train_ids, train_masks, train_labels = encode_tweets(
        tweets_text=X_train,
        tweets_labels=y_train,
        roberta_tokenizer=tokenizer
    )
    validation_ids, validation_masks, val_labels = encode_tweets(
        tweets_text=X_val,
        tweets_labels=y_val,
        roberta_tokenizer=tokenizer
    )
    test_ids, test_masks, test_labels = encode_tweets(
        tweets_text=X_test,
        tweets_labels=y_test,
        roberta_tokenizer=tokenizer
    )

    train_data = generate_roberta_dict(train_ids, train_masks)
    validation_data = (generate_roberta_dict(validation_ids, validation_masks), val_labels)
    test_data = generate_roberta_dict(test_ids, test_masks)

    model = roberta_tuning()
    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=2, restore_best_weights=True)
    csv_logger = CSVLogger(os.path.join(VIDGEN_MODEL_LOGS_PATH, f"{MODEL_FILE_NAME}.log"), separator=",", append=False)
    start_time = time.time()
    hist = model.fit(train_data, train_labels, validation_data=validation_data, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     callbacks=[csv_logger, early_stop])
    end_time = time.time()
    model.save(os.path.join(VIDGEN_MODEL_PATH, f"{MODEL_FILE_NAME}.h5"))
    display_readable_time(start_time=start_time, end_time=end_time)

    log_data = pd.read_csv(os.path.join(VIDGEN_MODEL_LOGS_PATH, f"{MODEL_FILE_NAME}.log"), sep=",", engine="python")
    display_train_report_and_f1_score(log_data)
    plot_train_data(log_data, train_metric="accuracy", validation_metric="val_accuracy")
    plot_train_data(log_data, train_metric="loss", validation_metric="val_loss")
    plot_train_data(log_data, train_metric="precision", validation_metric="val_precision")
    plot_train_data(log_data, train_metric="recall", validation_metric="val_recall")
    plt.show()

    # # ======= Test Model =======
    # new_model = load_model(os.path.join(VIDGEN_MODEL_PATH, f"{MODEL_FILE_NAME}.h5"))
    # predictions = new_model.predict(X_test)
    predictions = model.predict(test_data)

    for prediction in predictions:
        for index, pred_class in enumerate(prediction):
            if pred_class > 0.5:
                prediction[index] = 1
            else:
                prediction[index] = 0

    print(predictions)
    print(test_labels)
    print(len(predictions), len(test_labels))
    print(type(test_labels), type(predictions))

    print(f"\n{classification_report(test_labels, predictions)}")
