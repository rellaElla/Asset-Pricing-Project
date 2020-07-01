# System Imports
import os
import sys
from pathlib import Path
import pickle

# NLP Imports
import nltk
nltk.download('stopwords')
import re
from emoji import demojize

# Misc. Libraries
import pandas as pd
import numpy as np
from time import time

# Misc. Data Cleaning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Model Creation Libraries 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorboard.plugins.hparams import api as hp

# Custom imports
from preprocess import preprocess

class SIMP(object):
    def __init__(self):
        pass

    def fetch_data(self, file_name: str, input_length: int = 100, save_tokenizer: bool = False, save_encoder = False):
        """
        Takes in tweets as a pd.Series. quiet=True will return time to clean data. Returns pd.Series. 
        Operations:
        1. All text lowercase
        2. Replace special characterize
        3. Remove emojis
        4. Remove repetitions
        5. Change contractions to full words
        6. Remove English stopwords
        7. Stem tweets 
        """
        sys.path.append(Path(os.path.join(os.path.abspath(''), '.../')).resolve().as_posix())

        self.input_length = input_length
        dataset_path: Path = Path(f'.../data/{file_name}').resolve()

        print('Retrieivng Data')
        df: pd.DataFrame = pd.read_csv(dataset_path, encoding='latin-1')
        print('Data Found: Beginning Preprocessing')
        df['text'] = preprocess(df['text'])

        tweets: pd.DataFrame = df.text.to_numpy()
        labels: pd.DataFrame = df.label.to_numpy()

        # Set number of words and tokenize
        self.MAX_LENGTH: int = 50000
        self.tokenizer: Tokenizer = Tokenizer(num_words=self.MAX_LENGTH, lower=True)
        self.tokenizer.fit_on_texts(tweets)
        
        print('Data Tokenized')

        # Save tokenized data to disk
        if save_tokenizer:
            file_to_save: Path = Path('.../ticker_data/sentiment_data/tokenizer.pickle').resolve()
            with file_to_save.open('wb') as file:
                pickle.dump(self.tokenizer, file)

        # Tokenize Tweets
        tweet_sequences: list = [tweet.split() for tweet in tweets]
        tokenized_tweets: list = self.tokenizer.texts_to_sequences(tweet_sequences)

        # Transform data to numpy array of shape (num_samples, num_timesteps)
        padded_tweets: np.ndarray = pad_sequences(tokenized_tweets, maxlen=input_length)

        # Binarize for 1vAll Classification
        encoder: LabelBinarizer = LabelBinarizer()
        self.unique_labels: int = np.unique(labels)
        encoder.fit(self.unique_labels)

        print('Data Encoded')

        # Save encoder to disk 
        if save_encoder:
            encoder_path: Path = Path('.../ticker_data/sentiment_data/encoder.pickle').resolve()
            with encoder_path.open('wb') as file:
                pickle.dump(encoder, file)

        encoded_labels: np.ndarray = encoder.transform(labels.astype(str))
        # Split into train and val
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(padded_tweets, encoded_labels)

        # self.X_train = X_train.reshape(-1,1)
        # self.y_train = y_train.reshape(-1,1)
        # self.X_val = X_val.reshape(-1,1)
        # self.y_val = y_val.reshape(-1,1)

    def _train_model(self, hparams):
        input_dim: int = min(self.tokenizer.num_words, len(self.tokenizer.word_index) + 1)
        num_classes:int  = len(self.unique_labels)

        # Create input and output layers for LSTM
        input_layer: tf.Tensor = Input(shape=(self.input_length,))

        output_layer: tf.Tensor = Embedding(input_dim=input_dim, 
                                            output_dim=hparams[self.HP_EMBEDDING_DIM],
                                            input_shape=(self.input_length,)
                                            )(input_layer)

        output_layer: tf.Tensor = SpatialDropout1D(hparams[self.HP_SPATIAL_DROPOUT])(output_layer)

        output_layer: tf.Tensor = Bidirectional(LSTM(hparams[self.HP_LSTM_UNITS], return_sequences=True, 
                                                        dropout=hparams[self.HP_LSTM_DROPOUT], 
                                                        recurrent_dropout=hparams[self.HP_RECURRENT_DROPOUT])
                                                        )(output_layer)
                                                        
        output_layer: tf.Tensor = Conv1D(hparams[self.HP_FILTERS], kernel_size=hparams[self.HP_KERNEL_SIZE], 
                                                    padding='valid',
                                                    kernel_initializer='glorot_uniform'
                                                    )(output_layer)

        # Create global and max pooling to merge features across timesteps 
        avg_pool: tf.Tensor = GlobalAveragePooling1D()(output_layer)
        max_pool: tf.Tensor = GlobalMaxPooling1D()(output_layer)
        output_layer: tf.Tensor = concatenate([avg_pool, max_pool])

        # Add final dense layer to predict classes (deeply connected layer)
        output_layer: tf.Tensor = Dense(num_classes, activation='softmax')(output_layer)

        # Initialize model 
        model: tf.keras.Model = Model(input_layer, output_layer)

        # Set batch size and number of epochs
        batch_size: int = 128
        epochs: int = 10 

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        # Fit model 
        model.fit(
                    self.X_train, 
                    y=self.y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(self.X_val, self.y_val)
        )

        _, accuracy = model.evaluate(self.X_val, self.y_val)

        return model, accuracy
    
    def tune_model(self, 
        HP_EMBEDDING_DIM: int = hp.HParam('embedding_dim', hp.Discrete([400, 500, 700])),
        HP_LSTM_UNITS: int = hp.HParam('lstm_units', hp.Discrete([128, 256, 512])),
        HP_LSTM_DROPOUT: float = hp.HParam('lstm_dropout', hp.RealInterval(0.1, 0.5)),
        HP_RECURRENT_DROPOUT: float = hp.HParam('recurrent_dropout', hp.RealInterval(0.1, 0.5)),
        HP_SPATIAL_DROPOUT: float = hp.HParam('spatial_dropout', hp.RealInterval(0.1, 0.5)),
        HP_FILTERS: int = hp.HParam('filters', hp.Discrete([64, 128, 256])),
        HP_KERNEL_SIZE: int = hp.HParam('kernel_size', hp.Discrete([3])),
        METRIC = 'accuracy'
    ):
        self.HP_EMBEDDING_DIM: int = HP_EMBEDDING_DIM
        self.HP_LSTM_UNITS: int = HP_LSTM_UNITS
        self.HP_LSTM_DROPOUT: float = HP_LSTM_DROPOUT
        self.HP_RECURRENT_DROPOUT: float = HP_RECURRENT_DROPOUT
        self.HP_SPATIAL_DROPOUT: float = HP_SPATIAL_DROPOUT
        self.HP_FILTERS: int = HP_FILTERS
        self.HP_KERNEL_SIZE: int = HP_KERNEL_SIZE
        self.METRIC: str = METRIC

        with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
            hp.hparams_config(
                hparams=[self.HP_EMBEDDING_DIM, 
                        self.HP_LSTM_UNITS, 
                        self.HP_LSTM_DROPOUT,
                        self.HP_RECURRENT_DROPOUT,
                        self.HP_SPATIAL_DROPOUT,
                        self.HP_FILTERS,
                        self.HP_KERNEL_SIZE],
                metrics=[hp.Metric(self.METRIC, display_name='Accuracy')],
            )

        # Function to run hparams testing
        def run(run_dir, hparams, best_run):
            with tf.summary.create_file_writer(run_dir).as_default():
                hp.hparams(hparams)  # record the values used in this trial
                model, metric = self._train_model(hparams)
                
                if metric > best_run:
                    # Save best model's parameters to disk 
                    model_file: Path = Path('.../ticker_data/models/model_weights.h5').resolve()
                    model.save_weights(model_file.as_posix())

                tf.summary.scalar(self.METRIC, metric, step=1)
        
            return metric

        best_metric = 0
        # Loops to iterate over all hyperparameters
        session_num = 0
        for embedding_dim in self.HP_EMBEDDING_DIM.domain.values:
            for lstm_units in self.HP_LSTM_UNITS.domain.values:
                for lstm_dropout in (self.HP_LSTM_DROPOUT.domain.min_value, self.HP_LSTM_DROPOUT.domain.max_value):
                    for recurrent_dropout in (self.HP_RECURRENT_DROPOUT.domain.min_value, self.HP_RECURRENT_DROPOUT.domain.max_value):
                        for spatial_dropout in (self.HP_SPATIAL_DROPOUT.domain.min_value, self.HP_RECURRENT_DROPOUT.domain.max_value):
                            for filt in self.HP_FILTERS.domain.values:
                                for kernel in self.HP_KERNEL_SIZE.domain.values:
                                    hparams = {
                                        self.HP_EMBEDDING_DIM: embedding_dim,
                                        self.HP_LSTM_UNITS: lstm_units,
                                        self.HP_LSTM_DROPOUT: lstm_dropout,
                                        self.HP_RECURRENT_DROPOUT: recurrent_dropout,
                                        self.HP_SPATIAL_DROPOUT: spatial_dropout,
                                        self.HP_FILTERS: filt,
                                        self.HP_KERNEL_SIZE:  kernel 
                                        }

                                    run_name = "run-%d" % session_num
                                    print('--- Starting trial: %s' % run_name)
                                    print({h.name: hparams[h] for h in hparams})
                                    met = run('logs/hparam_tuning/' + run_name, hparams, best_metric)

                                    if met > best_metric:
                                            best_metric = met
                                    session_num += 1


model = SIMP()
model.fetch_data('master_df.csv')
model.tune_model()





 




