"""
Logistic Regression Example: Salary Expectation Classification
Author: OpenAI assistant
Purpose:
    This script demonstrates a complete Logistic Regression workflow
    for classifying whether a student's salary expectation is high or not.

Target classes:
    1 = High salary expectation
    0 = Low / moderate salary expectation
"""

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. Create the dataset
# ------------------------------------------------------------
# Each row represents one student / graduate profile.
# High_Expectation is the target variable:
#   1 = high salary expectation
#   0 = not high
data = {
    'Age': [22, 23, 24, 25, 26, 27, 28, 24, 29, 23],
    'GPA': [3.2, 3.8, 3.9, 3.5, 3.7, 3.1, 3.9, 3.4, 3.6, 3.0],
    'Experience': [0, 1, 2, 2, 3, 4, 5, 1, 6, 0],
    'Skills': [3, 5, 6, 4, 7, 5, 8, 4, 9, 2],
    'Internship': [0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    'High_Expectation': [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

# Convert dictionary into a pandas DataFrame
df = pd.DataFrame(data)

print("Dataset:")
print(df)
print("\n" + "-" * 60)

# ------------------------------------------------------------
# 2. Define input features (X) and target (y)
# ------------------------------------------------------------
# X contains the predictor variables
# y contains the class label we want to predict
X = df[['Age', 'GPA', 'Experience', 'Skills', 'Internship']]
y = df['High_Expectation']

# ------------------------------------------------------------
# 3. Split the dataset into training and testing sets
# ------------------------------------------------------------
# test_size=0.2 means 20% of the data is used for testing
# random_state=42 ensures reproducible results
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("-" * 60)

# ------------------------------------------------------------
# 4. Scale the input features
# ------------------------------------------------------------
# StandardScaler standardizes features to have mean 0 and std 1
# This is important because Logistic Regression often performs better
# when all features are on a similar scale.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 5. Train the Logistic Regression model
# ------------------------------------------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# 6. Make predictions
# ------------------------------------------------------------
# y_pred gives the predicted class labels (0 or 1)
# y_prob gives the predicted probabilities for each class
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ------------------------------------------------------------
# 7. Evaluate the model
# ------------------------------------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("-" * 60)

# ------------------------------------------------------------
# 8. Confusion Matrix Visualization
# ------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "High"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ------------------------------------------------------------
# 9. ROC Curve Visualization
# ------------------------------------------------------------
# ROC-AUC measures how well the classifier separates the two classes
auc_score = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ------------------------------------------------------------
# 10. Coefficient Plot
# ------------------------------------------------------------
# Coefficients help us understand the contribution of each feature
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient")

plt.figure(figsize=(8, 5))
plt.barh(coef_df["Feature"], coef_df["Coefficient"])
plt.xlabel("Coefficient Value")
plt.title("Logistic Regression Feature Coefficients")
plt.show()

print("Model coefficients:")
print(coef_df)
print("-" * 60)

# ------------------------------------------------------------
# 11. Predict for a new student
# ------------------------------------------------------------
new_student = pd.DataFrame([[25, 3.8, 2, 6, 1]],
                           columns=['Age', 'GPA', 'Experience', 'Skills', 'Internship'])

# Scale the new student data using the same scaler from training
new_student_scaled = scaler.transform(new_student)

# Predict class and probability
prediction = model.predict(new_student_scaled)
probability = model.predict_proba(new_student_scaled)

print("Predicted class:", prediction[0])
print("Predicted probability:", probability)

# ------------------------------------------------------------
# 12. Simple 2D classroom plot (Age vs Skills)
# ------------------------------------------------------------
# This extra plot is just for visual classroom explanation.
# It is NOT the actual training space because the model uses all features.
plt.figure(figsize=(8, 6))
colors = ['red' if label == 0 else 'green' for label in y]
plt.scatter(df['Age'], df['Skills'], c=colors, s=80)
plt.xlabel("Age")
plt.ylabel("Skills")
plt.title("Class Distribution (Age vs Skills)")
plt.show()
