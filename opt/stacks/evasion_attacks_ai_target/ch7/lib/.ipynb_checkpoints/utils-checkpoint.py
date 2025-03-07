import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet_v2 import  decode_predictions
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import inspect

# Utility functions to load and show samples, and a wrapper to the model's predict fuction to simplify its output

model=None


def show_image(img):
    plt.imshow(img)
    plt.axis('off')   


def show_images(images, titles, cmap=None):
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.show()

def preprocess_image(img, preprocess=preprocess_input):
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array =  preprocess(img_array)  
    img_array = np.clip(img_array, 0, 1)
    
    return img_array


def predictions(x, top=1):
    # Apply adversarial examples on target model
    predictions = model.predict(x)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=top)
    return decoded_predictions

def predict(x):
    # Predict with ResNet50
    preds = model.predict(x)
    # Decode predictions
    label= np.argmax(preds, axis=1)[0]
    decoded_preds = decode_predictions(preds, top=5)[0]

    prediction = {
        "label": label,
        "wordnet_id": decoded_preds[0][0],
        "class_name": decoded_preds[0][1],
        "confidence_score": decoded_preds[0][2]
    }
    
    return prediction




def show_adversarial_images(sample, x_adv):
     # Calculate the perturbation
    perturbation = x_adv - sample
    # Scaling perturbation for visualization
    perturbation_display = perturbation / (2 * np.max(np.abs(perturbation))) + 0.5
    perturbation_img = keras_image.array_to_img(perturbation_display[0])
    original_img = keras_image.array_to_img(sample[0])
    adv_img = keras_image.array_to_img(x_adv[0])
    # Show images side by side
    show_images([original_img, perturbation_img, adv_img], ['Original Image', 'Perturbation', 'Adversarial Image'])
    print('prediction for original image: \n',predict(sample))
    print('prediction for adversarial image: \n',predict(x_adv))
    return adv_img


# Load and preprocess an example image
def load_preprocess(img_path, preprocess=preprocess_input, show=False):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    if show:
        show_image(img)
    x = preprocess_image(img)
    return x

