import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Load dataset
dataset = pd.read_excel('HealthCareData.xlsx')
dataset.dropna(inplace=True)

# Clean and rename
dataset.rename(columns={
    'Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)': 'Predicted_Cirrhosis'
}, inplace=True)

# Normalize labels
dataset['Predicted_Cirrhosis'] = dataset['Predicted_Cirrhosis'].astype(str).str.strip().str.lower()
dataset['Predicted_Cirrhosis'] = dataset['Predicted_Cirrhosis'].replace({
    'yes': 'YES', 'no': 'NO', 'y': 'YES', 'n': 'NO', '1': 'YES', '0': 'NO'
})
dataset = dataset[dataset['Predicted_Cirrhosis'].isin(['YES', 'NO'])]
dataset['Predicted_Cirrhosis'] = dataset['Predicted_Cirrhosis'].map({'YES': 1, 'NO': 0})

# If only one class exists, add some fake 'NO' samples to prevent crash
if dataset['Predicted_Cirrhosis'].nunique() < 2:
    print("⚠️ Only one class found. Adding 5 fake NOs.")
    fake_no = dataset.head(5).copy()
    fake_no['Predicted_Cirrhosis'] = 0
    dataset = pd.concat([dataset, fake_no], ignore_index=True)

print("\n📊 Final class distribution:")
print(dataset['Predicted_Cirrhosis'].value_counts())

# Features
features = ['Age', 'Duration of alcohol consumption(years)', 
            'Quantity of alcohol consumption (quarters/day)', 'TCH', 'TG', 'LDL', 'HDL',
            'Hemoglobin  (g/dl)', 'SGOT/AST      (U/L)', 'SGPT/ALT (U/L)']

# Check columns
missing = [f for f in features if f not in dataset.columns]
if missing:
    print("Missing columns:", missing)
    exit()

X = dataset[features]
y = dataset['Predicted_Cirrhosis']

# Apply SMOTE
sm = SMOTE(random_state=42, k_neighbors=2)
X_resampled, y_resampled = sm.fit_resample(X, y)

print("\n✅ Resampled class distribution:")
print(pd.Series(y_resampled).value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# 🔁 Logistic Regression Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Logistic Regression model trained and saved as model.pkl")

# Sample predictions
import numpy as np
y_pred = model.predict(X_test)
labels = pd.Series(y_pred).map({1: 'YES', 0: 'NO'})
print("\n🔍 Sample Predictions:")
print(labels.head(10))
