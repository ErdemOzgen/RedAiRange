from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('simple-cifar10.h5')
cifar10_class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

app = Flask(__name__)

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
    image = file.read()
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

@app.route('/predict_multiple', methods=['POST'])
def predict_multiple():
    data = request.get_json()

    if 'images' not in data:
        return jsonify({'error': 'no images provided'}), 400

    images = data['images']

    if not isinstance(images, list):
        return jsonify({'error': 'images must be a list'}), 400

    # Preprocess the images
    processed_images = []
    for image in images:
        image = np.array(image).reshape(32, 32, 3)
        image = image.astype('float32') / 255
        processed_images.append(image)

    processed_images = np.array(processed_images)

    # Make the prediction
    predictions = model.predict(processed_images)
    label_indices = np.argmax(predictions, axis=1)

    # Create list of labels and class names
    response = []
    for label_index in label_indices:
        label = int(label_index)
        class_name = cifar10_class_names[label_index]
        response.append({
            'label': label,
            'class_name': class_name
        })

    return jsonify(response)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
