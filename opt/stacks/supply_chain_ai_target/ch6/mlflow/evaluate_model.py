import mlflow 
import mlflow.keras 
import tensorflow as tf 

from mlflow.tracking import MlflowClient
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import mlflow.keras

# Set MLflow Tracking URI l
mlflow.set_tracking_uri("http://127.0.0.1:5000") 

# Initialize the MlflowClient
client = MlflowClient()
# Get the latest version of the model
model_version = client.get_latest_versions("EnhancedCIFAR10_CNN_Model")[0]

(x_train, y_train _), (x_test, y_test) = cifar10.load_data()
# Normalize pixel values to [0, 1]
x_test =  x_test / 255.0

# Retrieve the model from MLflow
model_uri = "models:/EnhancedCIFAR10_CNN_Model/latest"
model = mlflow.keras.load_model(model_uri)
...
# Evaluate clean accuracy
loss, clean_accuracy = model.evaluate(x_test, to_categorical(y_test))
print(loss,clean_accuracy)
# Evaluate adversarial accuracy
#let s hard code the result here for the shake of the example to demo the flow
adv_accuracy = 0.3
client.set_model_version_tag(model_version.name, model_version.version, "clean_accuracy", clean_accuracy)
client.set_model_version_tag(model_version.name, model_version.version, "clean_accuracy", clean_accuracy)

# Update model tag based on evaluation
if clean_accuracy > 0.7 and adv_accuracy > 0.4:
    print("Model Tagged as Safe")
    client.set_model_version_tag(model_version.name, model_version.version, "safety", "safe")
else:
    print("Model Tagged as failing the test (tested-failed)")
    client.set_model_version_tag(model_version.name, model_version.version, "safety", "tested-failed")