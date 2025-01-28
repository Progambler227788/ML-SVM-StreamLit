# Muhammad Talha Atif
# Importing correct libraries to be used

import pandas as pd
import numpy as np
import joblib

# for cleaning the data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split # 70-30 split krny k leye
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Load dataset
df = pd.read_csv('User_Data.csv')

# Initialize LabelEncoder and StandardScaler
encode = LabelEncoder()
scaling = StandardScaler()

# Fit and transform the Gender column using LabelEncoder
df['Gender'] = encode.fit_transform(df['Gender'])  # 1 --> male, 0 --> female

# Fit and transform the Age and EstimatedSalary columns using StandardScaler
df[['Age', 'EstimatedSalary']] = scaling.fit_transform(df[['Age', 'EstimatedSalary']])

# Save LabelEncoder and StandardScaler as .pkl files
joblib.dump(encode, 'label_encoder.pkl')
joblib.dump(scaling, 'scaler.pkl')

# Now, we can proceed with the rest of the code for training the model

# Split data into features (X) and target (Y)
X, Y = df[['Gender', 'Age', 'EstimatedSalary']], df['Purchased']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=34)  # 70-30 split

# Initialize and fit the SVC model
svm_model = SVC()
svm_model.fit(X_train, Y_train)

# Save the trained SVM model
joblib.dump(svm_model, 'talha_atif_svm_model.pkl')

print("LabelEncoder, Scaler, and SVM Model have been saved successfully!")
