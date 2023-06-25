# Imports
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from flask import Flask, request, session

import time
from helpers import CustomModel, DataTransformer, ModelEvaluation

# Create web server app
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Set a secret key for session

# Transform data
model_name = 'aubmindlab/bert-base-arabertv02'
transformer = DataTransformer(model_name=model_name)

# Hyperparameters
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
custom_model = CustomModel(OPTIMIZER=OPTIMIZER, LOSS=LOSS, METRICS=METRICS)
model = custom_model.create_model(VOCAB_SIZE=VOCAB_SIZE, EMBEDDING_DIM=EMBEDDING_DIM, MAX_LEN=MAX_LEN,
                                  LSTM_SIZE=LSTM_SIZE, HIDDEN_SIZE=HIDDEN_SIZE, NUM_CLASSES=NUM_CLASSES)
model.load_weights('models/name_verification_model.h5')

# Model Evaluation
evaluator = ModelEvaluation()

# Define routes
@app.route("/", methods=['GET', 'POST'])
def index():
    if 'results' not in session:
        session['results'] = []

    if request.method == 'POST':
        input_data = request.form['input']
        start_time = time.time()
        X_test = transformer.transform_data(input_data, None)
        y_pred = evaluator.predictions(model, X_test)
        pred = 'Correct' if y_pred['Correct'][0] == 1 else 'Incorrect'
        output = f'The Full Name is: <span style="color: {"green" if pred == "Correct" else "red"};">{pred}</span>'
        end_time = time.time() - start_time
        execution_time = f'Execution Time is: {end_time:.2f} seconds'
        result = {'input': input_data, 'output': output, 'execution_time': execution_time}
        session['results'].append(result)

    return """
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        
        h1 {
            font-size: 24px;
        }
        
        form {
            margin-bottom: 20px;
        }
        
        input[type="text"] {
            padding: 5px;
            font-size: 16px;
            width: 300px;
        }
        
        input[type="submit"] {
            padding: 5px 10px;
            font-size: 16px;
            background-color: #007bff;
            color: #fff;
            border: none;
            cursor: pointer;
        }
        
        ul {
            list-style-type: none;
            padding: 0;
        }
        
        li {
            margin-bottom: 10px;
        }
        
        strong {
            font-weight: bold;
        }
    </style>
    <script>
        // Check if session storage exists
        if (typeof(Storage) !== "undefined") {
            // Clear session storage when the page is loaded
            sessionStorage.clear();
            
            // Clear session storage when the tab is closed
            window.addEventListener('beforeunload', function() {
                sessionStorage.clear();
            });
        }
    </script>
    <h1>Enter Full Name</h1>
    <form action="/" method="post">
        <input type="text" name="input" placeholder="Enter Full Name">
        <input type="submit" value="Submit">
    </form>
    <h2>Results</h2>
    <ul>
        """ + "".join([f"""
        <li>
            <strong>Input:</strong> {result['input']}<br>
            <strong>Output:</strong> {result['output']}<br>
            <strong>Execution Time:</strong> {result['execution_time']}<br>
        </li>
        <hr>
        """ for result in session['results']]) + """
    </ul>
    """

# Run the app
if __name__ == "__main__":
    app.run()
