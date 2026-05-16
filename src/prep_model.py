import mlflow
import argparse
from pathlib import Path
from mlflow.tracking import MlflowClient
from optimum.exporters.onnx import main_export


mlflow.set_tracking_uri("http://localhost:5012")

client = MlflowClient()

MODEL_PATH = "models/distilbert/"

parser = argparse.ArgumentParser(
    prog='Prep Model',
    description='Pepare Model for usage in API. Downloads model and exports it into onnx format.'
)

parser.add_argument("-m", '--model', required=True)
parser.add_argument("-v", '--version', required=True)

args = parser.parse_args()

base_uri = client.get_model_version_download_uri(
    args.model, version=args.version
)

print(f"Downloading model from: {base_uri}")

mlflow.artifacts.download_artifacts(
    artifact_uri=f"{base_uri}/components/tokenizer",
    dst_path=MODEL_PATH
)

mlflow.artifacts.download_artifacts(
    artifact_uri=f"{base_uri}/model",
    dst_path=MODEL_PATH
)

print(f"Exporting model to onnx format...")

main_export(
    model_name_or_path=str(Path(f"{MODEL_PATH}/model").resolve()),
    output=Path(f"{MODEL_PATH}/onnx/"),
    task="text-classification",
)
