from flask import Flask, jsonify, request

from predictor import Predictor
app = Flask(__name__)

predictor_instance = Predictor()
predictor_instance.init()
@app.route("/predict-message", methods=['POST'])
def predict_message():
    json = request.get_json()
    print(json)
    text = predictor_instance.predict(json['text'])
    result = {
        'text': text
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run()