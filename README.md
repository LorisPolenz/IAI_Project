# IAI Project 1

## Setup

### API
To run the API first download or train a model to serve. At the moment the API is only compatible with distilbert models. 

For the API to work download the follwing ZIP and unpack it into `models/`. You need to have a `models/distilbert/onnx` and `models/distilbert/tokenizer` directory with it's contents. 

To train and use the supported model (distilbert) follow the steps in the [Distilbert](#Distilbert) secion.


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

#### ML FLow
All the following steps require MLFlow First create the `.env` in `.mlflow` and then start it with
```bash
cd .mlflow && docker compose up -d 
```

MLflow is now available under `http://localhost:5012`

### Distilbert
> This step requires ML Flow to be setup

After executing the following commands the notebook to fine-tune distilbert can be used. 

```bash
python3.13 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

With this run the notebook in `src/fine_tune.ipynb`. 

To use the model with the API it needs to be downloaded from the artifact store and converted to the onnx format. This can be done with the following command:

```python
python3 src/prep_model.py --model "distilbert-finetuned" --version '1'
```
> Adapt the model and version to your parameters in MLFLow

### TF_IDF
> This step requires ML Flow to be setup

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


