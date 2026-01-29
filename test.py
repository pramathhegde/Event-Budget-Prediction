

from src.preprocessing import load_and_preprocess_data

X, y, encoders = load_and_preprocess_data("C:/Users/prama/Downloads/historical_event_budgets.csv")

print(X.head())
print(y.head())
