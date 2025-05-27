import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset
df = pd.read_csv(r"C:\Users\valan\OneDrive\Desktop\Projects in D Science\MANDATORY\metadata1.csv")

# Define selected numerical features
selected_numeric_cols = ['feat_hair', 'feat_asymmetry', 'feat_border_irregularity', 'feat_color']

# Fill missing values for selected numerical features using MEDIAN
df[selected_numeric_cols] = df[selected_numeric_cols].apply(lambda x: x.fillna(x.median()))

# Convert `biopsed` to numeric (1 = True, 0 = False)
df['biopsed'] = df['biopsed'].astype(int)

# Select only the specified features
X = df[selected_numeric_cols].copy()
y = df['biopsed']

# Standardize the numerical features
scaler = StandardScaler()
X[selected_numeric_cols] = scaler.fit_transform(X[selected_numeric_cols])

### **TRAINING LOGISTIC REGRESSION MODEL** ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate Accuracy, Precision, and Recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)

print(f"\nModel Accuracy: {accuracy:.2f}")
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
