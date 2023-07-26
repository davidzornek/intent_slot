from jsonschema import validate
import flask

from model import TokenClassifierModel
from config import API_CONFIG#, API_SCHEMA

app = flask.Flask(__name__)

model = TokenClassifierModel(
    API_CONFIG["base_model"],
    API_CONFIG["num_labels"],
    API_CONFIG["learning_rate"],
)


@app.route('/predict', methods=["POST"])
def predict():
    """
    Test with:
        curl -X POST http://127.0.0.1:5000/predict
            -H 'Content-Type: application/json'
            -d '{"text": "This is a test."}'
    """
    if flask.request.method != 'POST':
        return {"errors": ["Only POSTs supported at this time."]}

    try:
        req = flask.request.get_json()

    except Exception as e:
        return {"errors": [e]}

    result = model.predict(text_list=[req["text"]])

    # try:
    #     validate(req, schema=API_SCHEMA)
    # except Exception as e:
    #     return {"errors": [e]}

    return {
        "text": req["text"],
        "prediction": result["prediction"]
    }

if __name__ == "__main__":
    app.run()
