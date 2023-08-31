# intent_slot
This is a work in progress to create code for serving transformer-based intent-slot models.

- `docker build -t my_api .`
- `docker run -it -p 5000:5000 --name my_api my_api`
- Prediction endpoint: `127.0.0.1:5000/predict`
- Request format: `{text: "This is the text to predict on."}`
 
