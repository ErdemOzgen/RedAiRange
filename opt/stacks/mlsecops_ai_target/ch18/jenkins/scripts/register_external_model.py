
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from lib.model_registry import ModelRegistry
from lib.files import create_sha256
import argparse
import mlflow
import datetime

def timestamped_string(string):
  print(string)
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  return string + "-" + timestamp


def register_model(tracking_uri, model_file, model_name, description="", source="", dataset_name="", dataset_version="",  run="Jenkins Pipeline"):
    mlflow.set_tracking_uri(tracking_uri)
    if model_name == "":
        model_name = model_file.split("/")[-1].split(".")[0]
    hash = create_sha256(model_file)
    pipeline = timestamped_string(run)
    with mlflow.start_run() as run:
        reg = ModelRegistry()
        # Log tags relevant to both the model and the experiment
        reg.registration_tags["av scanned"] = True
        reg.registration_tags["model Scanned"] = True
        reg.registration_tags["mlflow.runName"] = pipeline
        for key in reg.registration_tags:
            mlflow.set_tag(key, reg.registration_tags[key])     
            print(f"Tag '{key}' set to '{reg.registration_tags[key]}'.")
        # tags only relevant to the model
        reg.registration_tags["hash"] = hash
        reg.registration_tags["stage"] = "evaluation"
        reg.registration_tags["registered by"] =pipeline
        del reg.registration_tags["mlflow.runName"]
        reg.log_model_dynamically(model_file,model_name, alias="scanned", description=description)        
        print(f"Model '{model_name}' registered in MLflow Model Registry.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register a model with MLflow and log additional experiment attributes.')
    parser.add_argument('--tracking-uri', required=True, help='The MLflow tracking URI.')
    parser.add_argument('--model-file', required=True, help='The file path of the model to register.')
    parser.add_argument('--run', required=False, help='The name of the pipeline.')
    parser.add_argument('--source', required=False, help='The source of the model.')
    parser.add_argument('--dataset-name', required=False, help='The name of the dataset used.')
    parser.add_argument('--dataset-version', required=False, help='The version of the dataset used.')
    parser.add_argument('--model-name', required=False, help='The name to register the model as in the MLflow Model Registry.')
    args = parser.parse_args()
    print("Arguments")
    print(args)
    register_model( args.tracking_uri, args.model_file, args.model_name,run=args.run) 
    
