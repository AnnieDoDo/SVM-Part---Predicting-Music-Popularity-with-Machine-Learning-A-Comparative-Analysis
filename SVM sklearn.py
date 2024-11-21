import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# Load the dataset
from google.colab import drive
drive.mount('/content/drive')
url = '/content/drive/MyDrive/dataset.csv'

dataset = pd.read_csv(url)
columns_to_preserve = ['artists', 'tempo', 'instrumentalness', 'popularity', 'speechiness', 'valence', 'danceability', 'album_name', 'acousticness', 'track_genre']
column_data_types = {'Unnamed': 'int', 'track_id': 'str', 'artists': 'str', 'album_name':'str', 'track_name':'str', 'popularity':'int', 'duration_ms': 'int', 'explicit': 'bool', 'danceability': 'float', 'energy':'float', 'key':'float', 'loudness':'float', 'mode':'int', 'speechiness':'float', 'acousticness':'float', 'instrumentalness':'float', 'liveness':'float', 'valence':'float', 'tempo':'float', 'time_signature': 'int', 'track_genre': 'str'}
synthetic_data = pd.read_csv(url, dtype=column_data_types)

features = synthetic_data[['danceability','energy','key','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']].values
target = synthetic_data['track_genre'].values

unique_labels = np.unique(target)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
encoded_target = np.array([label_mapping[label] for label in target])
encoded_target = np.where(encoded_target == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(features, encoded_target, test_size=0.2, random_state=42)

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def initialize_weights(self, n_features):
        return np.zeros(n_features)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = self.initialize_weights(n_features)
        self.bias = 0

        for iteration in range(1, self.n_iterations + 1):
            margins = y * (X.dot(self.weights) - self.bias)
            hinge_loss = np.maximum(0, 1 - margins)

            grad_weights = -X.T.dot(y * (margins <= 1))
            grad_bias = -np.sum(y * (margins <= 1))

            self.weights -= self.lr * (grad_weights + 2 * self.lambda_param * self.weights)
            self.bias -= self.lr * grad_bias


    def predict(self, X):
        return np.sign(X.dot(self.weights) - self.bias)

svm_model = SVM(learning_rate=0.01, lambda_param=0.01, n_iterations=1000)
svm_model.fit(X_train, y_train)

model = SVC()
accuracy_scores = cross_val_score(model, features, encoded_target, cv=5)

mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

print(f"Mean Accuracy: {mean_accuracy}")
print(f"Standard Deviation: {std_accuracy}")
# Use the SVM model for predictions on the test set
y_pred = svm_model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Test Accuracy:", accuracy)

mse = np.round(np.mean((y_test - y_pred) ** 2), 5)
mae = np.round(np.mean(np.abs(y_test - y_pred)), 5)
mape = np.round(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100, 5)

start = datetime.now()
metric_output = {
    'mean_squared_error': mse,
    'mean_absolute_error': mae,
    'mean_absolute_percentage_error': mape,
    'time': (datetime.now() - start).seconds
}

print(metric_output)
