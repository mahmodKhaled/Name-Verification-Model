import pandas as pd
import sklearn as sk
# Load test data
test_data = pd.read_csv('test_data.csv')
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

# Load trained model
model = YourModelClass()
model.load_model('trained_model.pth')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = sk.metrics.accuracy_score(y_test, y_pred)
precision = sk.metrics.precision_score(y_test, y_pred)
recall = sk.metrics.recall_score(y_test, y_pred)

# Print performance metrics
print("Performance Metrics:")
print("----------------------------")
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("----------------------------")
