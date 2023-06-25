# Imports
from fastapi import FastAPI
import time

# Create web server app
app = FastAPI()

# Load the model tokenizer and preprocessor
model_name = 'aubmindlab/bert-base-arabertv02'
model = model_init()

# Get requests
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