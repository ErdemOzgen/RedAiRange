import argparse
import mlflow

def quarantine_model(tracking_uri, model_file):
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run() as run:
        mlflow.set_tag('mlflow.runName', 'Quarantined')
        mlflow.log_param('model_file', model_file)
        print(f"Model {model_file} registered with tag 'Quarantined'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quarantine a model in MLflow.')
    parser.add_argument('--tracking-uri', required=True, help='The MLflow tracking URI.')
    parser.add_argument('--model-file', required=True, help='The file path of the model to quarantine.')
    
    args = parser.parse_args()
    
    quarantine_model(args.tracking_uri, args.model_file)
