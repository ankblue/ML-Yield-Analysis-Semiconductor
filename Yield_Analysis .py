import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from collections import Counter
import time

# Load UCI SECOM dataset
df = pd.read_csv("uci-secom.csv")  # Update with actual dataset path

# Data Preprocessing
# Drop non-numeric columns (e.g., TIME)
if 'Time' in df.columns:
    df = df.drop(columns=['Time'])

# Fill missing values with mean
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# Standardize features
scaler = StandardScaler()
X = df.drop(columns=['Pass/Fail'])  # Replace 'Pass/Fail' with actual label column name
y = df['Pass/Fail']
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to handle class imbalance
print("Before SMOTE:", Counter(y))
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("After SMOTE:", Counter(y_resampled))

# Feature Selection using PCA
print("Starting PCA...")
pca = PCA(n_components=0.90)  # Reduce to 90% variance
X_pca = pca.fit_transform(X_resampled)
print("PCA completed!")

# Train-Test Split
print("Starting Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_resampled, test_size=0.2, random_state=42)
print("Train-Test Split completed!")

# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [50, 100, 200],  # Reduced for faster training
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("Starting RandomForest Training...")
start_time = time.time()
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
end_time = time.time()
print(f"RandomForest Training completed in {end_time - start_time:.2f} seconds!")
best_rf_model = grid_search.best_estimator_

# Predictions
y_pred = best_rf_model.predict(X_test)

# Evaluate Model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-Validation Score
cv_scores = cross_val_score(best_rf_model, X_pca, y_resampled, cv=3)
print("Cross-Validation Accuracy:", np.mean(cv_scores))

# Feature Importance Analysis
importances = best_rf_model.feature_importances_
best_pc_index = np.argmax(importances)  # Identify the best principal component

plt.figure(figsize=(12, 6))
sns.barplot(x=np.arange(1, len(importances) + 1), y=importances)
plt.xlabel("Principal Component")
plt.ylabel("Feature Importance")
plt.title("Feature Importance in Yield Classification")
plt.xticks(np.arange(1, len(importances) + 1, step=5), rotation=90)  # Show every 5th label for better readability
plt.show()

# Identify Original Features Contribution to Best Principal Component
original_feature_contributions = pd.DataFrame(pca.components_.T, index=X.columns, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
print(f"Top Original Features for Best Principal Component (PC{best_pc_index + 1}):")
print(original_feature_contributions[f'PC{best_pc_index + 1}'].abs().sort_values(ascending=False).head(10))

# Visualizing Explained Variance of PCA
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA')
plt.show()
