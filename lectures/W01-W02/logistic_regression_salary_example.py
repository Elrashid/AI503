import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create dataset
data = {
    'Age': [22, 23, 24, 25, 26, 27, 28, 24, 29, 23],
    'GPA': [3.2, 3.8, 3.9, 3.5, 3.7, 3.1, 3.9, 3.4, 3.6, 3.0],
    'Experience': [0, 1, 2, 2, 3, 4, 5, 1, 6, 0],
    'Skills': [3, 5, 6, 4, 7, 5, 8, 4, 9, 2],
    'Internship': [0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    'High_Expectation': [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Inputs and output
X = df[['Age', 'GPA', 'Experience', 'Skills', 'Internship']]
y = df['High_Expectation']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# New example
new_student = pd.DataFrame([[25, 3.8, 2, 6, 1]],
                           columns=['Age', 'GPA', 'Experience', 'Skills', 'Internship'])

new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)
probability = model.predict_proba(new_student_scaled)

print("Predicted class:", prediction[0])
print("Predicted probability:", probability)
