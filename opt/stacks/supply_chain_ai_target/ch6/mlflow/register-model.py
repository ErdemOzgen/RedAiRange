import mlflow 
import mlflow.keras 
import tensorflow as tf 
import os
import requests 
from mlflow.tracking import MlflowClient

# Set MLflow Tracking URI l
mlflow.set_tracking_uri("http://127.0.0.1:5000") 
# Make sure the quarantine_area directory exists
if not os.path.exists('quarantine_area'):
    os.makedirs('quarantine_area')
# Download the model from Hugging Face via HTTPS URL 
model_url = " https://huggingface.co/DeepCyber/Enhanced-CIFAR10-CNN/resolve/main/enhanced-cif10-cnn.h5" 
model_path = "quarantine_area/enhanced-cif10-cnn.h5" 
r = requests.get(model_url) 
with open(model_path, 'wb') as f: 
    f.write(r.content) 

# Load the model 
model = tf.keras.models.load_model(model_path) 

# Log and register the model in MLflow, tagging it as "unsafe" 

with mlflow.start_run() as run: 
    mlflow.keras.log_model(model, "model") 
    mlflow.set_tag("safety", "unsafe") 
    mlflow.set_tag("status", "untested") 
model_uri = f"runs:/{run.info.run_id}/model" 
model_details = mlflow.register_model(model_uri, "EnhancedCIFAR10_CNN_Model") 
# Initialize the MlflowClient
client = MlflowClient()
# Set tags for the model version
client.set_registered_model_tag(model_details.name, "safety", "unsafe")
client.set_model_version_tag(model_details.name, model_details.version, "status", "untested")
client.set_model_version_tag(model_details.name, model_details.version, "stage", "evaluation")