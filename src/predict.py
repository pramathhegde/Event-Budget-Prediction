import joblib
import pandas as pd
import os

# Correct project root (budget-prediction-module)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pkl")

# Load trained model
model = joblib.load(model_path)


def predict_budget(event_type, participants, duration, department):
    input_data = pd.DataFrame([{
        "event_type": event_type,
        "participants": participants,
        "duration": duration,
        "department": department
    }])

    predicted_budget = model.predict(input_data)[0]
    return round(predicted_budget, 2)


if __name__ == "__main__":
    print(
        predict_budget(
            event_type=2,
            participants=150,
            duration=2,
            department=1
        )
    )
