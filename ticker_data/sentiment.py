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


def preprocess(texts, quiet=False):
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
  start: time = time()

  # Lowercasing
  texts: pd.Series = texts.str.lower()

  # Remove special chars
  texts = texts.str.replace(r"(http|@)\S+", "")
  texts = texts.apply(demojize)
  texts = texts.str.replace(r"::", ": :")
  texts = texts.str.replace(r"’", "'")
  texts = texts.str.replace(r"[^a-z\':_]", " ")

  # Remove repetitions
  pattern: re.SRE_Pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
  texts = texts.str.replace(pattern, r"\1")

  # Transform short negation form
  texts = texts.str.replace(r"(can't|cannot)", 'can not')
  texts = texts.str.replace(r"n't", ' not')

  # Remove stop words
  stopwords: nltk.stem = nltk.corpus.stopwords.words('english')

  stopwords.remove('not')
  stopwords.remove('nor')
  stopwords.remove('no')
  texts = texts.apply(
    lambda x: ' '.join([word for word in x.split() if word not in stopwords])
    )

  # Stem Tweets
  stemmer = nltk.stem.SnowballStemmer('english')
  texts = texts.apply(
    lambda x: ''.join([stemmer.stem(word) for word in x])
    )

  if not quiet:
    print("Time to clean up: {:.2f} sec".format(time() - start))

  return texts

class Dataset(object):
  """
  Attributes:
  filename: str
  label_col: str
  text_col: str
  dataframe: pd.DataFrame
  Properties:
  data: returns raw data
  cleaned_data: returns cleaned data
  Methods:
  load: fetches csv file using filename
  preprocess_texts: calls preprocess on self.dataframe
  """
  def __init__(self, filename, label_col='label', text_col='text'):
    self.filename: str = filename
    self.label_col: str = label_col
    self.text_col: str = text_col

  @property
  def data(self):
    data: pd.DataFrame = self.dataframe[[self.label_col, self.text_col]].copy()
    data.columns = ['label', 'text']
    return data

  @property
  def cleaned_data(self):
    data: pd.DataFrame =  self.dataframe[[self.label_col, 'cleaned']]
    data.columns = ['label', 'text']
    return data

  def load(self):
    df: pd.DataFrame = pd.read_csv(Path(self.filename).resolve(), encoding='latin-1')
    self.dataframe = df

  def preprocess_texts(self, quiet=False):
    self.dataframe['cleaned'] = preprocess(self.dataframe[self.text_col], quiet)

def train_test_model(hparams):
  print('Retrieivng Data')
  x_train, y_train, x_validation, y_validation, unique_vals = datawork()

  file_to_open: Path = Path('.../ticker_data/sentiment_data/tokenizer.pickle').resolve()
  with file_to_open.open('rb') as file:
      tokenizer = pickle.load(file)

  # Set model hyperparameters/dimensions
  input_dim: int = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
  num_classes:int  = len(unique_vals)
  input_length: int = 100


  #embedding_dim: int = 500
  #lstm_units: int = 256
  #lstm_dropout: float = 0.15
  #recurrent_dropout: float = 0.1
  #spatial_dropout: float = 0.3
  #filters: int = 128
  #kernel_size:int = 3

  # Create input and output layers for LSTM
  input_layer: tf.Tensor = Input(shape=(input_length,))

  output_layer: tf.Tensor = Embedding(input_dim=input_dim, 
                                      output_dim=hparams[HP_EMBEDDING_DIM],
                                      input_shape=(input_length,)
                                      )(input_layer)

  output_layer: tf.Tensor = SpatialDropout1D(hparams[HP_SPATIAL_DROPOUT])(output_layer)

  output_layer: tf.Tensor = Bidirectional(LSTM(hparams[HP_LSTM_UNITS], return_sequences=True, 
                                                dropout=hparams[HP_LSTM_DROPOUT], 
                                                recurrent_dropout=hparams[HP_RECURRENT_DROPOUT])
                                                )(output_layer)
                                                
  output_layer: tf.Tensor = Conv1D(hparams[HP_FILTERS], kernel_size=hparams[HP_KERNEL_SIZE], 
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
            x_train, 
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_validation, y_validation)
  )

  _, accuracy = model.evaluate(x_validation, y_validation)

  return model, accuracy

def datawork():
  input_length: int = 100
  # Set path
  sys.path.append(Path(os.path.join(os.path.abspath(''), '.../')).resolve().as_posix())

  # Find dataset and preprocess it 
  dataset_path: Path = Path('.../ticker_data/sentiment_data/aapl_sentiment.csv').resolve()
  dataset: Dataset = Dataset(dataset_path)
  dataset.load()
  dataset.preprocess_texts()

  # Set number of words and tokenize
  num_words: int = 50000
  tokenizer: Tokenizer = Tokenizer(num_words=num_words, lower=True)
  tokenizer.fit_on_texts(dataset.cleaned_data.text)

  # Save tokenized data to disk
  file_to_save: Path = Path('.../ticker_data/sentiment_data/tokenizer.pickle').resolve()
  with file_to_save.open('wb') as file:
      pickle.dump(tokenizer, file)

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

  # Generate train and validation samples
  train_sequences: list = [text.split() for text in train.text]
  validation_sequences: list = [text.split() for text in validation.text]

  # Tokenize these lists 
  list_tokenized_train: list = tokenizer.texts_to_sequences(train_sequences)
  list_tokenized_validation: list = tokenizer.texts_to_sequences(validation_sequences)

  # Transform data to numpy array of shape (num_samples, num_timesteps)
  x_train: np.ndarray = pad_sequences(list_tokenized_train, maxlen=input_length)
  x_validation: np.ndarray = pad_sequences(list_tokenized_validation, maxlen=input_length)

  # Does 1vAll Binarization 
  encoder: LabelBinarizer = LabelBinarizer()
  encoder.fit(data.label.unique())

  # Save encoder to disk 
  encoder_path: Path = Path('.../ticker_data/sentiment_data/encoder.pickle').resolve()
  with encoder_path.open('wb') as file:
      pickle.dump(encoder, file)

  # Get transformed values for 1vAll classification 
  y_train: np.ndarray = encoder.transform(train.label.values.astype(str))
  y_validation: np.ndarray = encoder.transform(validation.label.values.astype(str))

  return x_train, y_train, x_validation, y_validation, data.label.unique()

# Tuneable Hyperparameters
HP_EMBEDDING_DIM: int = hp.HParam('embedding_dim', hp.Discrete([400, 500, 700]))
HP_LSTM_UNITS: int = hp.HParam('lstm_units', hp.Discrete([128, 256, 512]))
HP_LSTM_DROPOUT: float = hp.HParam('lstm_dropout', hp.RealInterval(0.1, 0.5))
HP_RECURRENT_DROPOUT: float = hp.HParam('recurrent_dropout', hp.RealInterval(0.1, 0.5))
HP_SPATIAL_DROPOUT: float = hp.HParam('spatial_dropout', hp.RealInterval(0.1, 0.5))
HP_FILTERS: int = hp.HParam('filters', hp.Discrete([64, 128, 256]))
HP_KERNEL_SIZE: int = hp.HParam('kernel_size', hp.Discrete([3]))

# Metric we'll use 
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_EMBEDDING_DIM, 
              HP_LSTM_UNITS, 
              HP_LSTM_DROPOUT,
              HP_RECURRENT_DROPOUT,
              HP_SPATIAL_DROPOUT,
              HP_FILTERS,
              HP_KERNEL_SIZE],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )


# Function to run hparams testing
def run(run_dir, hparams, best_run):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    model, accuracy = train_test_model(hparams)
    
    if accuracy > best_run:
      # Save best model's parameters to disk 
      model_file: Path = Path('.../ticker_data/models/model_weights.h5').resolve()
      model.save_weights(model_file.as_posix())


    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
  
  return accuracy

best_accuracy = 0
# Loops to iterate over all hyperparameters
session_num = 0
for embedding_dim in HP_EMBEDDING_DIM.domain.values:
  for lstm_units in HP_LSTM_UNITS.domain.values:
    for lstm_dropout in (HP_LSTM_DROPOUT.domain.min_value, HP_LSTM_DROPOUT.domain.max_value):
      for recurrent_dropout in (HP_RECURRENT_DROPOUT.domain.min_value, HP_RECURRENT_DROPOUT.domain.max_value):
        for spatial_dropout in (HP_SPATIAL_DROPOUT.domain.min_value, HP_RECURRENT_DROPOUT.domain.max_value):
          for filt in HP_FILTERS.domain.values:
            for kernel in HP_KERNEL_SIZE.domain.values:
                  hparams = {
                    HP_EMBEDDING_DIM: embedding_dim,
                    HP_LSTM_UNITS: lstm_units,
                    HP_LSTM_DROPOUT: lstm_dropout,
                    HP_RECURRENT_DROPOUT: recurrent_dropout,
                    HP_SPATIAL_DROPOUT: spatial_dropout,
                    HP_FILTERS: filt,
                    HP_KERNEL_SIZE:  kernel 
                    }

                  run_name = "run-%d" % session_num
                  print('--- Starting trial: %s' % run_name)
                  print({h.name: hparams[h] for h in hparams})
                  acc = run('logs/hparam_tuning/' + run_name, hparams, best_accuracy)

                  if acc > best_accuracy:
                        best_accuracy = acc
                  session_num += 1



