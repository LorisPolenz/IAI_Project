import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient
from optimum.exporters.onnx import main_export


mlflow.set_tracking_uri("http://localhost:5012")

client = MlflowClient()

MODEL_PATH = "models/distilbert/"

base_uri = client.get_model_version_download_uri(
    "distilbert-finetuned",
    version="12"
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
