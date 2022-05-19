import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.metrics import Precision, Recall
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from transformers import XLNetTokenizer, TFXLNetModel, PreTrainedTokenizerBase

from src.utils import process_data, plot_train_data, display_train_report_and_f1_score, display_readable_time


MODEL_FILE_NAME = "xlnet_model"
XLNET_TYPE = "xlnet-base-cased"  # xlnet-large-cased | xlnet-base-cased
MAX_PADDING_LENGTH = 28
LEARNING_RATE = 2e-5
THRESHOLD = 0.85
BATCH_SIZE = 2
EPOCHS = 10


def encode_tweet(tweet: str, xlnet_tokenizer: PreTrainedTokenizerBase):
    return xlnet_tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length=MAX_PADDING_LENGTH,
        truncation=True,
        padding="max_length",
        return_attention_mask=True
    )


def encode_tweets(tweets_text: list, tweets_labels, xlnet_tokenizer: PreTrainedTokenizerBase):
    if tweets_labels is not None:
        assert len(tweets_text) == len(tweets_labels), f"Features and labels must have the same lengths. " \
                                                       f"Your input ({len(tweets_text)}, {len(tweets_labels)})"

    input_ids = []
    token_type_ids = []
    attention_masks = []

    for tweet in tweets_text:
        tweet_for_xlnet = encode_tweet(tweet, xlnet_tokenizer)
        input_ids.append(tweet_for_xlnet["input_ids"])
        token_type_ids.append(tweet_for_xlnet["token_type_ids"])
        attention_masks.append(tweet_for_xlnet["attention_mask"])

    if tweets_labels is not None:
        assert len(input_ids) == len(token_type_ids) == len(attention_masks) == len(tweets_labels), \
            "Arrays must have the same length."
        return np.array(input_ids), np.array(token_type_ids), np.array(attention_masks), np.array(tweets_labels)

    return np.array(input_ids), np.array(token_type_ids), np.array(attention_masks)


def generate_xlnet_dict(input_ids, token_type_ids, attention_mask) -> dict:
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask
    }


def xlnet_tuning(xlnet_type: str = XLNET_TYPE):
    input_ids = tf.keras.Input(shape=(MAX_PADDING_LENGTH,), name="input_ids", dtype="int32")
    token_type_ids = tf.keras.Input(shape=(MAX_PADDING_LENGTH,), name="token_type_ids", dtype="int32")
    attention_masks = tf.keras.Input(shape=(MAX_PADDING_LENGTH,), name="attention_mask", dtype="int32")

    xlnet_model = TFXLNetModel.from_pretrained(xlnet_type)
    encodings = xlnet_model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)[0]
    last_encoding = tf.squeeze(encodings[:, -1:, :], axis=1)
    last_encoding = tf.keras.layers.Dropout(0.1)(last_encoding)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="outputs")(last_encoding)

    temp_model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_masks], outputs=[outputs])
    temp_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall()]
    )

    return temp_model


tokenizer = XLNetTokenizer.from_pretrained(XLNET_TYPE, do_lower_case=True)
train_df = pd.read_csv("../../../../Homework/nlp-disaster-tweets/data/train.csv")
test_df = pd.read_csv("../../../../Homework/nlp-disaster-tweets/data/test.csv")
train_text = train_df["text"].tolist()
train_res = train_df["target"].tolist()
test_text = test_df["text"].tolist()

for index, temp_tweet in enumerate(train_text):
    train_text[index] = process_data(temp_tweet)

for index, temp_tweet in enumerate(test_text):
    test_text[index] = process_data(temp_tweet)

X_train, X_val, y_train, y_val = train_test_split(train_text, train_res, test_size=0.2)

train_ids, train_tokens, train_masks, train_labels = encode_tweets(
    tweets_text=X_train,
    tweets_labels=y_train,
    xlnet_tokenizer=tokenizer
)
validation_ids, validation_tokens, validation_masks, validation_labels = encode_tweets(
    tweets_text=X_val,
    tweets_labels=y_val,
    xlnet_tokenizer=tokenizer
)
test_ids, test_tokens, test_masks = encode_tweets(
    tweets_text=test_text,
    tweets_labels=None,
    xlnet_tokenizer=tokenizer
)

train_data = generate_xlnet_dict(train_ids, train_tokens, train_masks)
validation_data = (generate_xlnet_dict(validation_ids, validation_tokens, validation_masks), validation_labels)
test_data = generate_xlnet_dict(test_ids, test_tokens, test_masks)

# model = xlnet_tuning()
# print(model.summary())
# csv_logger = CSVLogger(f"../models/logs/{MODEL_FILE_NAME}.log", separator=",", append=False)
# start_time = time.time()
# hist = model.fit(train_data, train_labels, validation_data=validation_data, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[csv_logger])
# end_time = time.time()
# model.save(f"../models/{MODEL_FILE_NAME}_10epochs.h5")
# display_readable_time(start_time=start_time, end_time=end_time)

# log_data = pd.read_csv(f"../models/logs/{MODEL_FILE_NAME}.log", sep=",", engine="python")
# display_train_report_and_f1_score(log_data)
# plot_train_data(log_data, train_metric="accuracy", validation_metric="val_accuracy")
# plot_train_data(log_data, train_metric="loss", validation_metric="val_loss")
# plot_train_data(log_data, train_metric="precision", validation_metric="val_precision")
# plot_train_data(log_data, train_metric="recall", validation_metric="val_recall")
# plt.show()

# 1 -> 10epochs model (0.9: 0.79926, 0.65: 0.79834)
# 2 -> 5epochs model (
new_model = load_model(f"../models/{MODEL_FILE_NAME}_5epochs.h5", custom_objects={'TFXLNetModel': TFXLNetModel})
sample_submission = pd.read_csv("../../../../Homework/nlp-disaster-tweets/data/sample_submission.csv", dtype=object)
predictions = new_model.predict(test_data)

for index, prediction in enumerate(predictions):
    # print(prediction, end=" | ")
    predictions[index] = 1 if prediction > THRESHOLD else 0

predictions = predictions.astype(int)
sample_submission["target"] = predictions
sample_submission.to_csv("../results/submission_xlnet3.csv", index=False)
