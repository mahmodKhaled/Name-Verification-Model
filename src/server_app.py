# Imports
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from fastapi import FastAPI
import time
from helpers import CustomModel, DataTransformer, ModelEvaluation

# Create web server app
app = FastAPI()

# Transform data
model_name = 'aubmindlab/bert-base-arabertv02'
transformer = DataTransformer(model_name=model_name)

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

# Model Evaluation
evaluator = ModelEvaluation()

# Get requests
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/API/Name-Verification")
def Inference(input: str):
  start_time = time.time()
  X_test = transformer.transform_data(input, None)
  y_pred = evaluator.predictions(model, X_test)
  pred = 'Correct' if y_pred['Correct'][0] == 1 else 'Incorrect'
  output = f'The Full Name is: {pred}'
  end_time = time.time() - start_time
  execution_time = f'Execution Time is: {end_time:.2f} seconds'
  return output, execution_time