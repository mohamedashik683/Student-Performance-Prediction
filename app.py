import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Page settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# Load dataset
data = pd.read_csv("student_data.csv")

features = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignment_Marks",
    "Internal_Marks",
    "Sleep_Hours"
]

X = data[features]
y = data["Final_Marks"]

# Train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
test_prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, test_prediction)
r2 = r2_score(y_test, test_prediction)

# Title
st.title("🎓 Student Performance Prediction")
st.write(
    "Machine Learning based system to predict student final marks."
)

st.divider()

# Input section
st.subheader("📝 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        "Study Hours", 0.0, 15.0, 5.0
    )

    attendance = st.number_input(
        "Attendance (%)", 0.0, 100.0, 85.0
    )

    previous_marks = st.number_input(
        "Previous Marks", 0.0, 100.0, 75.0
    )

with col2:
    assignment_marks = st.number_input(
        "Assignment Marks", 0.0, 100.0, 80.0
    )

    internal_marks = st.number_input(
        "Internal Marks", 0.0, 100.0, 78.0
    )

    sleep_hours = st.number_input(
        "Sleep Hours", 0.0, 12.0, 7.0
    )

# Prediction
if st.button("🔮 Predict Performance"):

    student = pd.DataFrame(
        [[
            study_hours,
            attendance,
            previous_marks,
            assignment_marks,
            internal_marks,
            sleep_hours
        ]],
        columns=features
    )

    prediction = model.predict(student)[0]
    prediction = max(0, min(100, prediction))

    st.divider()

    st.subheader("🎯 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predicted Final Marks",
            f"{prediction:.2f}"
        )

    with result_col2:
        if prediction >= 75:
            level = "High 🟢"
        elif prediction >= 50:
            level = "Average 🟡"
        else:
            level = "Low 🔴"

        st.metric("Performance Level", level)

    # Recommendations
    st.subheader("💡 Recommendations")

    if study_hours < 5:
        st.write("📚 Increase your study hours.")
    elif attendance < 75:
        st.write("🏫 Try to improve your attendance.")
    elif assignment_marks < 70:
        st.write("✍️ Improve your assignment performance.")
    elif internal_marks < 70:
        st.write("📖 Prepare well for internal examinations.")
    elif sleep_hours < 6:
        st.write("😴 Get enough sleep for better concentration.")
    else:
        st.write(
            "🎉 Good performance! Keep maintaining your current habits."
        )

# Data analysis
st.divider()
st.subheader("📊 Student Performance Analysis")

chart_data = data[
    ["Study_Hours", "Attendance", "Previous_Marks", "Final_Marks"]
]

st.write("Study Hours vs Final Marks")
st.scatter_chart(
    chart_data,
    x="Study_Hours",
    y="Final_Marks"
)

st.write("Attendance vs Final Marks")
st.scatter_chart(
    chart_data,
    x="Attendance",
    y="Final_Marks"
)

st.write("Previous Marks vs Final Marks")
st.scatter_chart(
    chart_data,
    x="Previous_Marks",
    y="Final_Marks"
)

# Model evaluation
st.divider()
st.subheader("🤖 Machine Learning Model Performance")

metric1, metric2 = st.columns(2)

with metric1:
    st.metric(
        "Mean Absolute Error",
        f"{mae:.2f}"
    )

with metric2:
    st.metric(
        "R² Score",
        f"{r2:.2f}"
    )

st.success("✅ Machine Learning model trained successfully!")