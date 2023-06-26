import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from helpers import CustomModel, DataTransformer, ModelEvaluation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import sys

# Load test data
test_data = pd.read_csv('input/processed_data/test_data.csv')
X_test = test_data.drop(columns=['Label'], axis=1)
y_test = test_data[['Label']]

# Transform data
model_name = 'aubmindlab/bert-base-arabertv02'
transformer = DataTransformer(model_name=model_name)
X_test, y_test = transformer.transform_data(X_test, y_test)

# hyperparameters
VOCAB_SIZE = 10000
MAX_LEN = 50
EMBEDDING_DIM = 16
HIDDEN_SIZE = 32
NUM_CLASSES = 2
BATCH_SIZE = 512
LSTM_SIZE = 16
OPTIMIZER = Adam()
LOSS = BinaryCrossentropy()
METRICS = ['accuracy']

# Load trained model
custom_model = CustomModel(OPTIMIZER= OPTIMIZER, LOSS= LOSS, METRICS= METRICS)
model = custom_model.create_model(VOCAB_SIZE= VOCAB_SIZE, EMBEDDING_DIM= EMBEDDING_DIM, MAX_LEN= MAX_LEN, 
                                LSTM_SIZE= LSTM_SIZE, HIDDEN_SIZE= HIDDEN_SIZE, NUM_CLASSES= NUM_CLASSES)
model.load_weights('models/name_verification_model.h5')

# Make predictions
evaluator = ModelEvaluation()
y_pred = evaluator.predictions(model, X_test)
y_pred = pd.DataFrame(y_pred, columns=['Correct','Incorrect'])

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Print performance metrics
print("Performance Metrics:")
print("----------------------------")
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("----------------------------")

# Send metrics as outputs
sys.stdout.write(f"::set-output name=accuracy::{accuracy}")
sys.stdout.write(f"::set-output name=precision::{precision}")
sys.stdout.write(f"::set-output name=recall::{recall}")
