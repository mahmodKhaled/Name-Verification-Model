# important imports
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from transformers import AutoTokenizer
from arabert.preprocess import ArabertPreprocessor
from fastapi import FastAPI
import time

# initialize the model
def model_init():
    # hyperparameters
    VOCAB_SIZE = 10000
    MAX_LEN = 50
    EMBEDDING_DIM = 16
    HIDDEN_SIZE = 32
    LSTM_SIZE = 16
    NUM_CLASSES = 2
    # architecture
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))
    model.add(Bidirectional(LSTM(LSTM_SIZE, return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(HIDDEN_SIZE, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.load_weights('name_verification_model.h5')
    return model

# transform data
def transform_data(X,y):
    """
    This function is used to transform the data into a format that can be used by the model.
    It will convert the names into ids and pad them to a length of 50.
    It will also encode the labels into one-hot encoding.
    Parameters:
    - X: A Pandas DataFrame containing the names.
    - y: A Pandas DataFrame containing the labels.
    Returns:
    - X: A numpy array containing the names in ids format.
    - y: A numpy array containing the labels in one-hot encoding format.
    """
    if not isinstance(X, pd.DataFrame):
        X = {'Name': [X]}
        X = pd.DataFrame(X, index=[0], columns=['Name'])
    def split_word(word):
        word = word.replace(" ", "")
        return list(word)
    # split each name into a list of characters
    X['Name'] = X['Name'].apply(split_word)
    # convert tokens in X to ids
    X['Name'] = X['Name'].apply(lambda x: tokenizer.convert_tokens_to_ids(x))
    # convert into numpy array
    X = pad_sequences(X['Name'], maxlen=50, padding='post', truncating='post')
    if y is None:
        return X 
    else:
        # encode labels in y data
        y = pd.get_dummies(y['Label'])
    return X, y

# predictions data
def predictions(model, X_test):
  """
    Makes predictions using a model.

    Parameters:
    - model: A model object.
    - X_test: A NumPy array containing the input data to use for predictions.

    Returns:
    - A dictionary containing the predictions. The dictionary has the following structure:
    {
      'Correct': list of correct predictions (0s and 1s),
      'Incorrect': list of incorrect predictions (0s and 1s)
    }
  """
  # perform predictions
  pred = model.predict(X_test)
  # this variable means that the model predicts the full name is correct and its confidence is higher than 
  # the confidence of the full name being incorrect
  correct_pred = (pred[:, 0] > pred[:, 1]).astype(int)
  # this variable means that the model predicts the full name is incorrect and its confidence is higher than 
  # the confidence of the full name being correct
  incorrect_pred = (pred[:, 1] > pred[:, 0]).astype(int)
  # put predictions into a dictionary under two keys correct and incorrect
  return {'Correct': correct_pred, 'Incorrect': incorrect_pred}
# create web server app
app = FastAPI()

# load the model tokenizer and preprocessor
model_name = 'aubmindlab/bert-base-arabertv02'
tokenizer =  AutoTokenizer.from_pretrained(model_name)
preprocessor = ArabertPreprocessor(model_name=model_name)
model = model_init()

# get requests
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/API/Name-Verification")
def Inference(input: str):
  start_time = time.time()
  X_test = transform_data(input, None)
  y_pred = predictions(model, X_test)
  pred = 'Correct' if y_pred['Correct'][0] == 1 else 'Incorrect'
  output = f'The Full Name is: {pred}'
  end_time = time.time() - start_time
  execution_time = f'Execution Time is: {end_time:.2f} seconds'
  return output, execution_time