# 1. Import necessary libraries
import pandas as pd                    # For working with data
import numpy as np                     # For numerical operations
import matplotlib.pyplot as plt        # For plotting charts
import seaborn as sns                  # For prettier plots
import os                              # For directory operations
from datetime import datetime          # For timestamping results

from sklearn.model_selection import train_test_split  # To split data
from sklearn.linear_model import LogisticRegression   # Our ML model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler      # For feature scaling

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# 2. Load the Heart Disease dataset directly from URL
url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
df = pd.read_csv(url)

# 3. Preview the dataset
print("🔍 First 5 rows of the dataset:")
print(df.head())

print("\n📊 Dataset shape:")
print(df.shape)

print("\n🧼 Checking for missing values:")
print(df.isnull().sum())

# 4. Separate features (X) and target (y)
X = df.drop('target', axis=1)  # Features (all columns except target)
y = df['target']               # Target variable (1 = disease, 0 = no disease)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 5. Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\n✅ Data split completed:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 6. Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)  # max_iter increased for convergence
model.fit(X_train, y_train)

# 7. Make predictions on the test set
y_pred = model.predict(X_test)

# 8. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy:.2f}")

# 9. Classification Report
print("\n📋 Classification Report:")
class_report = classification_report(y_test, y_pred)
print(class_report)

# 10. Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("💡 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Save the confusion matrix plot
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'results/confusion_matrix_{timestamp}.png')
plt.close()

# 11. Feature Importance Analysis
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Heart Disease Prediction')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.savefig(f'results/feature_importance_{timestamp}.png')
plt.close()

# 12. Save model metrics to file
with open(f'results/model_metrics_{timestamp}.txt', 'w') as f:
    f.write("Heart Disease Prediction Model Results\n")
    f.write("=====================================\n\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Model: Logistic Regression\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(class_report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nFeature Importance:\n")
    f.write(feature_importance.to_string())

print(f"\n✅ Results have been saved to the 'results' directory!")
