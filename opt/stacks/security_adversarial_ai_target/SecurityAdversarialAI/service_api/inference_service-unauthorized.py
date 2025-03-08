from flask import Flask, request, jsonify
import logging
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import os

# Disable GPU for inference
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('deployed_models/simple-cifar10.h5')
cifar10_class_names = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

app = Flask(__name__)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/heartbeat', methods=['GET'])
def heartbeat():
    return jsonify({'system':'api','status': 'OK'}), 200

# Function to preprocess image
def preprocess_image(image):
    image = Image.open(io.BytesIO(image))
    image = image.resize((32, 32))
    image = np.array(image)
    image = image.astype('float32') / 255  # normalization
    image = np.expand_dims(image, axis=0)  # expand dimension
    return image

@app.route('/predict', methods=['POST'])
def predict_single():
    # Get the image from the POST request
    file = request.files['file']
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    logger.info(f"Filename: {file.filename}")
    logger.info(f"Content-Type: {file.content_type}")
    file.seek(0)
    image = file.read()
    logger.info(f"File size: {len(image)} bytes")    
    image = preprocess_image(image)
    prediction = model.predict(image)
    label_index = np.argmax(prediction)
    # Get the label and class name
    label = int(label_index)
    class_name = cifar10_class_names[label_index]

    # Create and send response
    response = {
        'prediction': {
            'label': label,
            'class_name': class_name
        }
    }


    return jsonify(response)

    return jsonify(response)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
