# Imports

from tensorflow.keras.layers import Dense , Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from transformers import AutoTokenizer
from tensorflow.keras.utils import pad_sequences
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Custom Model Class
class CustomModel():
    def __init__(self, OPTIMIZER, LOSS, METRICS):
        self.OPTIMIZER = OPTIMIZER
        self.LOSS = LOSS
        self.METRICS = METRICS

    def create_model(self, VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, LSTM_SIZE, HIDDEN_SIZE, NUM_CLASSES):
        model = Sequential()
        model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))
        model.add(Bidirectional(LSTM(LSTM_SIZE, return_sequences=True)))
        model.add(GlobalMaxPool1D())
        model.add(Dense(HIDDEN_SIZE, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        return model

# Data Transformer Class
class DataTransformer:
    def __init__(self, model_name):
        self.tokenizer =  AutoTokenizer.from_pretrained(model_name)

    def transform_data(self, X,y):
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
        X['Name'] = X['Name'].apply(lambda x: self.tokenizer.convert_tokens_to_ids(x))
        # convert into numpy array
        X = pad_sequences(X['Name'], maxlen=50, padding='post', truncating='post')
        if y is None:
            return X 
        else:
            # encode labels in y data
            y = pd.get_dummies(y['Label'])
        return X, y

# Model Evaluation Class
class ModelEvaluation:
    def __init__(self):
        pass

    def plot_curves(self, history):
        """
        Plots the loss and accuracy curves for a model.

        Parameters:
        - history: A history object returned by a model during training.

        Returns:
        - None. The function displays the plot using Matplotlib.
        """
        title_loss = 'Model loss per epoch '
        title_accuracy = 'Model accuracy per epoch '
        fig , axis = plt.subplots(nrows=1, ncols=2)
        # dimensions of figure
        fig.set_figheight(6)
        fig.set_figwidth(14)
        # loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        # accuracy
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        epoch = np.arange(150)
        # loss curve
        axis[0].plot(loss,label='Train')
        axis[0].plot(val_loss,label='Validation')
        axis[0].set_xlabel('epoch')
        axis[0].set_ylabel('loss')
        axis[0].set_title(title_loss)
        axis[0].legend()
        # accuracy curve
        axis[1].plot(accuracy, label='Train')
        axis[1].plot(val_accuracy, label='Validation')
        axis[1].set_xlabel('epoch')
        axis[1].set_ylabel('accuracy')
        axis[1].set_title(title_accuracy)
        axis[1].legend()

    def predictions(self, model, X_test):
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
