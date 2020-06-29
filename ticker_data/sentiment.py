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

# Set model hyperparameters/dimensions
input_dim: int = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
num_classes:int  = len(data.label.unique())
embedding_dim: int = 500
input_length: int = 100
lstm_units: int = 256
lstm_dropout: float = 0.15
recurrent_dropout: float = 0.1
spatial_dropout: float = 0.3
filters: int = 128
kernel_size:int = 3

# Create input and output layers for LSTM; output is a 1D Spatial Dropout 
input_layer: tf.Tensor = Input(shape=(input_length,))
output_layer: tf.Tensor = Embedding(input_dim=input_dim, 
                                    output_dim=embedding_dim,
                                    input_shape=(input_length,)
                                    )(input_layer)

output_layer: tf.Tensor = SpatialDropout1D(spatial_dropout)(output_layer)

output_layer: tf.Tensor = Bidirectional(LSTM(lstm_units, return_sequences=True, 
                                              dropout=lstm_dropout, 
                                              recurrent_dropout=recurrent_dropout)
                                              )(output_layer)
output_layer: tf.Tensor = Conv1D(filters, kernel_size=kernel_size, 
                                          padding='valid',
                                          kernel_initializer='glorot_uniform'
                                          )(output_layer)

# Create global and max pooling to merge features across timesteps 
avg_pool: tf.Tensor = GlobalAveragePooling1D()(output_layer)
max_pool: tf.Tensor = GlobalMaxPooling1D()(output_layer)
output_layer: tf.Tensor = concatenate([avg_pool, max_pool])

# Add final dense layer to predict classesb (deeply connected layer)
output_layer: tf.Tensor = Dense(num_classes, activation='softmax')(output_layer)

# Initialize model 
model: tf.keras.Model = Model(input_layer, output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

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

# Set batch size and number of epochs
batch_size: int = 128
epochs: int = 3

# Fit model 
model.fit(
          x_train, 
          y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_validation, y_validation)
)

# Save best model's parameters to disk 
model_file: Path = Path('.../ticker_data/models/model_weights.h5').resolve()
model.save_weights(model_file.as_posix())