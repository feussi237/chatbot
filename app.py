from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chat import get_responses
#import nltk
#nltk.download('punkt_tab')

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes

@app.post("/predict")
def predict():
    text = request.get_json().get("message")

    response=get_responses(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    #app.run(debug=True, port=5000)  # Run the Flask app on port 5000
    # Note: In production, you might want to set debug=False and use a production server like Gunicorn or uWSGI.