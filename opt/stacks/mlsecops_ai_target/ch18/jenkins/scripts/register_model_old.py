import argparse
import mlflow


def log_model_dynamically(model, model_type, model_name):
    """
    Logs a model to MLflow dynamically based on the model type.

    Args:
    - model: The model object to be logged.
    - model_type: A string indicating the type of the model (e.g., 'sklearn').
    - model_name: The name under which to log the model.
    """
    if model_type in model_logging_functions:
        log_function = model_logging_functions[model_type]
        log_function(model, model_name)
        print(f"Model of type '{model_type}' logged under '{model_name}'.")
    else:
        raise ValueError(f"Unsupported model type '{model_type}'.")


def register_model(tracking_uri, model_file, pipeline):
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run() as run:
        mlflow.set_tag('mlflow.runName', pipeline)
        mlflow.log_param('model_file', model_file)
        print(f"Model {model_file} registered with tag 'Success'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register a model with MLflow.')
    parser.add_argument('--tracking-uri', required=True, help='The MLflow tracking URI.')
    parser.add_argument('--model-file', required=True, help='The file path of the model to register.')
    
    args = parser.parse_args()
    
    register_model(args.tracking_uri, args.model_file)
