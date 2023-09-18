# intent_slot
This is a work in progress to create code for serving transformer-based intent-slot models.

- `docker build -t my_api .`
- `docker run -it -p 5000:5000 --name my_api my_api`
    - During dev: `docker run -v ~/Documents/Git\ Repos/intent_slot/intent_slot/:/work/intent_slot -it --name dev fake_api bash`
- Prediction endpoint: `127.0.0.1:5000/predict`
- Request format: `{text: "This is the text to predict on."}`
