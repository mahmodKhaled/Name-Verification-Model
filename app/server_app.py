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
  correct_pred = []
  incorrect_pred = []
  # perform predictions
  pred = model.predict(X_test)
  for x in pred:
    # this case means that the model predicts the full name is correct and its confidence is higher than the confidence of the full name being incorrect
    if x[0] > x[1]:
      correct_pred.append(1)
      incorrect_pred.append(0)
    # this case means that the model predicts the full name is incorrect and its confidence is higher than the confidence of the full name being correct
    elif x[1] > x[0]:
      incorrect_pred.append(1)
      correct_pred.append(0)
  # put predictions into a dictionary under two keys correct and incorrect
  y_pred = {'Correct': correct_pred, 'Incorrect': incorrect_pred}
  return y_pred
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
    if y_pred['Correct'][0] == 1:
        pred = 'Correct'
    else:
        pred = 'Incorrect'
    output = 'The Full Name is: ' + pred
    end_time = (time.time() - start_time)
    execution_time = 'Execution Time is: ' + str(end_time)
    return output , execution_time