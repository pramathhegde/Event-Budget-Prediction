import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocessing import load_and_preprocess_data


# Load data
X, y, encoders = load_and_preprocess_data("C:/Users/prama/Downloads/historical_event_budgets.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- Linear Regression --------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

print("Linear Regression:")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)

# -------- Random Forest --------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

print("\nRandom Forest Regressor:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)

# -------- Save best model --------
best_model = rf if rf_rmse < lr_rmse else lr
joblib.dump(best_model, "model.pkl")

print("\nBest model saved as model.pkl")
