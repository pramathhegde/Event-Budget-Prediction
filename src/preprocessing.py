import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Handle missing values (safe default)
    df = df.dropna()

    # Encode categorical columns
    categorical_cols = ["event_type", "department"]
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Features and target
    X = df[["event_type", "participants", "duration", "department"]]
    y = df["approved_budget"]

    return X, y, encoders
