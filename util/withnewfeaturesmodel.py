import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset
df = pd.read_csv(r"C:\Users\valan\OneDrive\Desktop\Projects in D Science\MANDATORY\metadata1.csv")

### **DATA PREPROCESSING** ###

# List of categorical features
binary_cols = ['smoke', 'drink', 'itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation']

# Updated list of numerical features including new columns
numeric_cols = ['diameter_1', 'diameter_2', 'age', 'feat_hair', 'feat_asymmetry', 'feat_border_irregularity', 'feat_color']

# Convert 'UNK' to NaN, then replace True/False with 1/0
df[binary_cols] = df[binary_cols].replace({'True': 1, 'False': 0, 'UNK': np.nan}).astype(float)

# Fill missing values in binary features with 0.5 (for initial test)
df[binary_cols] = df[binary_cols].fillna(0.5)

# Fill missing values for numerical features using MEDIAN (to prevent outliers from impacting the data)
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))

# Convert `biopsed` to numeric (1 = True, 0 = False)
df['biopsed'] = df['biopsed'].astype(int)

# Select features
features = binary_cols + numeric_cols
X = df[features].copy()
y = df['biopsed']

# Standardize the numerical features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

### **TRAINING LOGISTIC REGRESSION MODEL** ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate Accuracy, Precision, and Recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)

print(f"\nModel Accuracy (0.5 for missing binary features): {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

### **FEATURE IMPORTANCE** ###
feature_importance = np.exp(model.coef_[0])  # Convert to odds ratios
print("\nFeature Importance (Odds Ratios):")
for feature, importance in zip(X.columns, feature_importance):
    print(f"{feature}: {importance:.2f}")

### **CROSS-VALIDATION** ###
kf = KFold(n_splits=5, shuffle=True, random_state=None)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.2f}")
