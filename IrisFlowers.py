# 1. Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 2. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# 3. Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Build a classification model
knn_classifier = KNeighborsClassifier(n_neighbors=3)  # You can adjust n_neighbors as needed

# 6. Train the model on the training data
knn_classifier.fit(X_train, y_train)

# 7. Evaluate the model on the testing data
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
