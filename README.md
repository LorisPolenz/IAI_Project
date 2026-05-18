# IAI Project 1

## Setup

### ML FLow
For most steps MLFlow is required. First create the `.env` in `.mlflow` and then start it with
```bash
cd .mlflow && docekr compose up -d 
```

MLflow is now available under `http://localhost:5012`

### API
To run the API first download or train a model to serve. At the moment the API is only compatible with distilbert models. 

For the API to work download the follwing ZIP and unpack it into `models/`. You need to have a `models/distilbert/onnx` and `models/distilbert/tokenizer` directory with it's contents. 

This can be done with the following command:

```bash 
curl -L "https://fsn1.your-objectstorage.com/iai/models.tar" | tar -xv
```

Afterwards build and run the docker image:

```bash
docker build -t nlp-api . && docker run --rm -p 8000:8000 --name nlp nlp-api 
```

Test with 
```bash 
curl -XPOST localhost:8000/predict \
-H 'Content-Type: application/json' \
-d '{"text": "I cannot stand the actor in this move."}' 
```

```bash
curl -XPOST localhost:8000/predict \
-H 'Content-Type: application/json' \
-d '{"text": "I love this movie!"}'
```

### Everything else

### Distilbert fine-tuning

### TF_IDF
To train the IF_IDF Model, execute the folling commands

```bash
python -m spacy download en_core_web_sm
```

```bash
python3.13 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

The script `src/tf_idf.ipynb` can now be used. 


