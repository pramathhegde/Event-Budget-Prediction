📊 Event Budget Prediction System

This project is an end-to-end Machine Learning–based Event Budget Prediction system built using Python and Flask. It predicts the approved budget for an event based on historical data and provides a simple web interface for real-time interaction with the trained model.

🚀 Features

Predicts event budgets using trained machine learning models

Uses historical event data with features like event type, participants, duration, and department

Compares multiple regression models and selects the best-performing one

Exposes predictions through a Flask REST API

Includes a web-based UI for easy user interaction

Clean separation between training, prediction, and API layers

🧠 Machine Learning Workflow

Data preprocessing (handling missing values and encoding categorical features)

Model training using Linear Regression and Random Forest Regressor

Model evaluation using MAE and RMSE

Automatic selection and persistence of the best model

Real-time inference using the saved model

🌐 Web Interface

Built using Flask + HTML + JavaScript

User-friendly dropdowns for categorical inputs (Event Type, Department)

Duration is taken in days

Displays predicted budget instantly without reloading the page

🏗️ Project Structure
BUDGET-PREDICTION-MODULE/
├── api/            # Flask application
├── src/            # ML logic (training, preprocessing, prediction)
├── templates/      # HTML UI
├── model.pkl       # Trained ML model
├── notebooks/      # Data analysis notebooks
├── requirements.txt

⚙️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Flask

Joblib

HTML, JavaScript

▶️ How to Run

Install dependencies:

pip install -r requirements.txt


Train the model (if not already trained):

python src/train_model.py


Start the Flask server:

python api/app.py


Open in browser:

http://127.0.0.1:5000/ui

📌 Use Case

This system can be used by organizations or institutions to estimate event budgets efficiently, reduce manual planning effort, and make data-driven financial decisions.
