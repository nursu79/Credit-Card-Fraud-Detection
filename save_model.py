import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
credit_card_data = pd.read_csv("creditcard.csv")  # Make sure the dataset is in the same directory

# Data Preprocessing
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Undersampling (balances the dataset)
legit_sample = legit.sample(n=423)  # Adjust based on dataset size
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Splitting features & target
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Splitting into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate the model
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")

# Save the trained model
joblib.dump(model, 'logistic_model.pkl')
print("Model saved as 'logistic_model.pkl'")
