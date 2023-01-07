# important imports
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense , Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from transformers import AutoTokenizer
from arabert.preprocess import ArabertPreprocessor
from fastapi import FastAPI
from typing import Union
import time

# initialize the model
def model_init():
    # hyperparameters
    VOCAB_SIZE = 60000
    MAX_LEN = 3
    EMBEDDING_DIM = 16
    HIDDEN_SIZE = 32
    NUM_CLASSES = 2
    # architecture
    model = Sequential([
        Embedding(input_dim= VOCAB_SIZE, output_dim= EMBEDDING_DIM, input_length= MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(units= HIDDEN_SIZE, activation = 'relu'),
        Dense(units= NUM_CLASSES, activation='softmax')
    ])
    model.load_weights('name_verification_model.h5')
    return model

# transform data
def transform_data(X,y):
  """
    Transforms input data for use in a model.

    Parameters:
    - X: A Pandas DataFrame containing the input data. The DataFrame must have the following structure:
    {
      'Name': list of names (strings)
    }
    - y: (Optional) A Pandas DataFrame containing the labels for the input data. The DataFrame must have the following structure:
    {
      'Label': list of labels (strings)
    }

    Returns:
    - If `y` is not provided, returns the transformed `X` data as a NumPy array.
    - If `y` is provided, returns a tuple containing the transformed `X` data as a NumPy array and the transformed `y` data as a Pandas DataFrame. The DataFrame has the following structure:
    {
      'Correct': list of correct labels (0s and 1s),
      'Incorrect': list of incorrect labels (0s and 1s)
    }
  """
  if not isinstance(X, pd.DataFrame):
    X = {'Name': [X]}
    X = pd.DataFrame(X, index=[0], columns=['Name'])
  # apply preprocessor on the X data
  X = X['Name'].apply(lambda x: preprocessor.preprocess(x))
  # convert back to dataframe and reset_index
  X = pd.DataFrame(X,columns=['Name'])
  # tokenize X data
  X = [tokenizer.tokenize(name ,max_length=3, truncation=True) for name in X['Name'].tolist()]
  # convert tokens in X to ids
  X = [tokenizer.convert_tokens_to_ids(name) for name in X]
  # convert into numpy array
  X = np.array(X)
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

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/Inference")
def Inference(input: str):
  start_time = time.time()
  X_test = transform_data(input, None)
  y_pred = predictions(model, X_test)
  pred = 'Correct' if y_pred['Correct'][0] == 1 else 'Incorrect'
  output = f'The Full Name is: {pred}'
  end_time = time.time() - start_time
  execution_time = f'Execution Time is: {end_time:.2f} seconds'
  return output, execution_time