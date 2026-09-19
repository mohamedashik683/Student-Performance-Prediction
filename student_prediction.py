import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("student_data.csv")

# Input features
X = data[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignment_Marks",
        "Internal_Marks",
        "Sleep_Hours"
    ]
]

# Target value
y = data["Final_Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Student Performance Prediction")
print("--------------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Predict a new student
new_student = [[5, 85, 75, 80, 78, 7]]

predicted_mark = model.predict(new_student)

print("\nNew Student Details:")
print("Study Hours: 5")
print("Attendance: 85%")
print("Previous Marks: 75")
print("Assignment Marks: 80")
print("Internal Marks: 78")
print("Sleep Hours: 7")

print("\nPredicted Final Marks:",
      round(predicted_mark[0], 2))