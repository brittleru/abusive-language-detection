import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from keras.utils import np_utils
from keras.models import load_model
from keras.metrics import Precision, Recall
from sklearn.metrics import classification_report
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase, TFAutoModel, AutoTokenizer, TFRobertaModel

from src.utils import process_data, display_readable_time, display_train_report_and_f1_score, plot_train_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
DAVIDSON_DIR = os.path.join(DATA_DIR, "hatespeech-davidson")
DAVIDSON_DIR = os.path.join(DAVIDSON_DIR, "data")
DATASET_PATH = os.path.join(DAVIDSON_DIR, "labeled_data.csv")
MODEL_LOGS_PATH = os.path.join(BASE_DIR, "model-logs")
DAVIDSON_MODEL_LOGS_PATH = os.path.join(MODEL_LOGS_PATH, "davidson")
MODEL_PATH = os.path.join(BASE_DIR, "models")
DAVIDSON_MODEL_PATH = os.path.join(MODEL_PATH, "davidson")


MODEL_FILE_NAME = "bertweet_large_davidson"
BERTWEET_TYPE = "vinai/bertweet-large"  # vinai/bertweet-base | vinai/bertweet-large

# Clean: 36 | No lowercase: 29 | Lowercase: 28 | Lowercase & Stemming: 28 | Lowercase & Lemmas: 28
MAX_PADDING_LENGTH = 28
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 10


def encode_tweet(tweet: str, bertweet_tokenizer: PreTrainedTokenizerBase):
    return bertweet_tokenizer.encode_plus(
        # tweet,
        # process_data(tweet),
        process_data(tweet, do_stemming=False, do_lemmas=False, do_lowercase=True),
        add_special_tokens=True,
        max_length=MAX_PADDING_LENGTH,
        truncation=True,
        padding="max_length",
        return_attention_mask=True
    )


def encode_tweets(tweets_text: list, tweets_labels, bertweet_tokenizer: PreTrainedTokenizerBase):
    if tweets_labels is not None:
        assert len(tweets_text) == len(tweets_labels), f"Features and labels must have the same lengths. " \
                                                       f"Your input ({len(tweets_text)}, {len(tweets_labels)})"

    input_ids = []
    attention_masks = []

    for tweet in tweets_text:
        tweet_for_bertweet = encode_tweet(tweet, bertweet_tokenizer)
        input_ids.append(tweet_for_bertweet["input_ids"])
        attention_masks.append(tweet_for_bertweet["attention_mask"])

    if tweets_labels is not None:
        assert len(input_ids) == len(attention_masks) == len(tweets_labels), \
            "Arrays must have the same length."
        return np.array(input_ids), np.array(attention_masks), np.array(tweets_labels)

    return np.array(input_ids), np.array(attention_masks)


def generate_bertweet_dict(input_ids, attention_mask) -> dict:
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def bertweet_tuning(bertweet_type: str = BERTWEET_TYPE):
    input_ids = tf.keras.Input(shape=(MAX_PADDING_LENGTH,), name="input_ids", dtype="int32")
    attention_masks = tf.keras.Input(shape=(MAX_PADDING_LENGTH,), name="attention_mask", dtype="int32")

    bertweet_model = TFAutoModel.from_pretrained(bertweet_type)
    encodings = bertweet_model(input_ids=input_ids, attention_mask=attention_masks)[0]
    last_encoding = tf.squeeze(encodings[:, -1:, :], axis=1)
    # last_encoding = tf.keras.layers.Dropout(0.1)(last_encoding)

    outputs = tf.keras.layers.Dense(3, activation="softmax", name="outputs")(last_encoding)

    temp_model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=outputs)
    temp_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", Precision(), Recall()]
    )

    return temp_model


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BERTWEET_TYPE, do_lower_case=True)

    df = pd.read_csv(DATASET_PATH, delimiter=",")

    train_texts = df["tweet"].tolist()
    train_labels = df["class"].tolist()

    train_labels = np_utils.to_categorical(train_labels)

    X_train, X_temp, y_train, y_temp = train_test_split(train_texts, train_labels, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    train_ids, train_masks, train_labels = encode_tweets(
        tweets_text=X_train,
        tweets_labels=y_train,
        bertweet_tokenizer=tokenizer
    )
    validation_ids, validation_masks, val_labels = encode_tweets(
        tweets_text=X_val,
        tweets_labels=y_val,
        bertweet_tokenizer=tokenizer
    )
    test_ids, test_masks, test_labels = encode_tweets(
        tweets_text=X_test,
        tweets_labels=y_test,
        bertweet_tokenizer=tokenizer
    )

    train_data = generate_bertweet_dict(train_ids, train_masks)
    validation_data = (generate_bertweet_dict(validation_ids, validation_masks), val_labels)
    test_data = generate_bertweet_dict(test_ids, test_masks)

    model = bertweet_tuning()
    print(model.summary())
    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=2, restore_best_weights=True)
    csv_logger = CSVLogger(os.path.join(DAVIDSON_MODEL_LOGS_PATH, f"{MODEL_FILE_NAME}.log"), separator=",",
                           append=False)
    start_time = time.time()
    hist = model.fit(train_data, train_labels, validation_data=validation_data, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     callbacks=[csv_logger, early_stop])
    end_time = time.time()
    model.save(os.path.join(DAVIDSON_MODEL_PATH, f"{MODEL_FILE_NAME}.h5"))
    display_readable_time(start_time=start_time, end_time=end_time)

    log_data = pd.read_csv(os.path.join(DAVIDSON_MODEL_LOGS_PATH, f"{MODEL_FILE_NAME}.log"), sep=",", engine="python")
    display_train_report_and_f1_score(log_data)
    plot_train_data(log_data, train_metric="accuracy", validation_metric="val_accuracy")
    plot_train_data(log_data, train_metric="loss", validation_metric="val_loss")
    plot_train_data(log_data, train_metric="precision", validation_metric="val_precision")
    plot_train_data(log_data, train_metric="recall", validation_metric="val_recall")
    plt.show()

    # # ======= Test Model =======
    # new_model = load_model(os.path.join(DAVIDSON_MODEL_PATH, f"{MODEL_FILE_NAME}.h5"))

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
