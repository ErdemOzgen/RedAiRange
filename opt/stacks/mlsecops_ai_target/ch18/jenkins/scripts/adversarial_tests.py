import argparse
import mlflow

def test( model_file):
   print(f"testing {model_file}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quarantine a model in MLflow.')
    parser.add_argument('--tracking-uri', required=True, help='The MLflow tracking URI.')
    parser.add_argument('--model-file', required=True, help='The file path of the model to quarantine.')
    
    args = parser.parse_args()
    
    test(args.model_file)
