import sys
import os
from flask import Flask, request, jsonify, render_template

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_budget

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Budget Prediction API is running"

@app.route("/ui")
def ui():
    return render_template("budget.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    required_fields = ["event_type", "participants", "duration", "department"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    predicted_budget = predict_budget(
        event_type=data["event_type"],
        participants=data["participants"],
        duration=data["duration"],
        department=data["department"]
    )

    return jsonify({
        "predicted_budget": predicted_budget
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
