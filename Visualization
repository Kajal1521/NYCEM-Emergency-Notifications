# NYCEM-Emergency-Notifications
Messages sent with information about emergency events and important City services
import pandas as pd

# Load the dataset
url = "https://data.cityofnewyork.us/Public-Safety/NYCEM-Emergency-Notifications/8vv7-7wx3"
data = pd.read_csv(url)
# Display basic information about the dataset
print(data.info())

# Display some descriptive statistics
print(data.describe())

# Visualize data to identify patterns
# (Use Matplotlib, Seaborn, or any other preferred visualization library)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Assume 'X' contains features and 'y' contains labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Model 2: Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Model 3: Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Model 4: Neural Network
nn_model = MLPClassifier()
nn_model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

# Example for Decision Tree
dt_predictions = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, dt_predictions)
report = classification_report(y_test, dt_predictions)

print(f"Decision Tree Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
