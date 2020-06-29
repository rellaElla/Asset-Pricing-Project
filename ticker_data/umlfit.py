from transformers import BertTokenizer
from dataset import Dataset
from preprocess import preprocess

# Misc. Data Cleaning Libraries
from sklearn.model_selection import train_test_split

# System Imports
import os
import sys
from pathlib import Path
import pickle

import pandas as pd

import tensorflow as tf
from transformers import TFBertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences


def encode_train_test(file, batch_size = 128, shuffle_size = 10000):

    # Set path
    sys.path.append(Path(os.path.join(os.path.abspath(''), '.../')).resolve().as_posix())

    # Find dataset and preprocess it 
    dataset_path: Path = Path(f'.../ticker_data/sentiment_data/{file}').resolve()
    dataset: Dataset = Dataset(dataset_path)
    dataset.load()
    dataset.preprocess_texts()

    # Copy the cleaned dataset to prevent pandas issues
    data: pd.DataFrame = dataset.cleaned_data.copy()
    train: pd.DataFrame = pd.DataFrame(columns=['label', 'text'])
    validation: pd.DataFrame = pd.DataFrame(columns=['label', 'text'])

    # Retrieve indices and split data for each label
    for label in data.label.unique():
        label_data: pd.Index = data[data.label == label]
        train_data, validation_data = train_test_split(label_data, test_size=0.3)
        train: pd.DataFrame = pd.concat([train, train_data])
        validation: pd.DataFrame = pd.concat([validation, validation_data])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    train_tokenized = train['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens = True))
    val_tokenized = validation['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens = True))

    train_tokenized = train_tokenized.values
    val_tokenized = val_tokenized.values

    # # Save tokenized data to disk
    # file_to_save: Path = Path('.../ticker_data/sentiment_data/tokenizer.pickle').resolve()
    # with file_to_save.open('wb') as file:
    #     pickle.dump(tokenizer, file)


    y_train = tf.convert_to_tensor(train['label'].values.astype(float).reshape(-1,1))
    y_val = tf.convert_to_tensor(validation['label'].values.astype(float).reshape(-1,1))

        # Transform data to numpy array of shape (num_samples, num_timesteps)
    x_train: np.ndarray = pad_sequences(train_tokenized, maxlen=len(y_train))
    x_validation: np.ndarray = pad_sequences(val_tokenized, maxlen=len(y_val))

    # # Get transformed values for 1vAll classification 
    # y_train: np.ndarray = encoder.transform(train.label.values.astype(str))
    # y_validation: np.ndarray = encoder.transform(validation.label.values.astype(str))

    return tf.data.Dataset.from_tensor_slices((x_train, y_train)), tf.data.Dataset.from_tensor_slices((x_validation, y_val))


X_train, y_val = encode_train_test('aapl_sentiment.csv')

learning_rate = 2e-5
# we will do just 1 epoch for illustration, though multiple epochs might be better as long as we will not overfit the model
number_of_epochs = 1
# model initialization
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
# classifier Adam recommended
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
#loss = tf.keras.losses.Cate(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[metric])

bert_history = model.fit(X_train, epochs=number_of_epochs, validation_data=y_val)
