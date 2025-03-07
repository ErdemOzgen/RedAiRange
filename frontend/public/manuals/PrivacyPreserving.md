# Privacy-Preserving AI: A Detailed Handout
# Use 'privacy_preserving_ai_target' machine for this practice 


Privacy-preserving AI is a framework for designing data analytics, machine learning (ML), and AI systems that protect sensitive data while still enabling useful insights. In today’s data-driven world, these techniques are crucial to comply with regulations (e.g., the European GDPR) and minimize risks related to data breaches and unauthorized re-identification.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Privacy-Preserving AI vs. Privacy-Preserving ML](#privacy-preserving-ai-vs-ml)
3. [Simple Data Anonymization Techniques](#simple-data-anonymization-techniques)
   - [Hashing, Masking, and Obfuscation](#hashing-masking-and-obfuscation)
   - [Python Example](#python-example)
4. [Advanced Anonymization Techniques](#advanced-anonymization-techniques)
   - [K-Anonymity](#k-anonymity)
   - [Microaggregation and Related Techniques](#microaggregation-and-related-techniques)
   - [Tools and Libraries](#tools-and-libraries)
5. [Anonymization of Geolocation Data](#anonymization-of-geolocation-data)
   - [Techniques: Geographic Masking & Spatial Aggregation](#techniques-geographic-masking-and-spatial-aggregation)
   - [Code Example](#geolocation-code-example)
6. [Anonymizing Rich Media](#anonymizing-rich-media)
   - [Images](#images)
     - [Blurring, Pixelation, and Masking](#blurring-pixelation-and-masking)
     - [Data Perturbation](#data-perturbation)
   - [Audio and Video](#audio-and-video)
     - [Techniques and a Python Example](#audio-anonymization-code-example)
7. [Differential Privacy (DP)](#differential-privacy-dp)
   - [Concept and Application](#concept-and-application)
   - [Code Example with TensorFlow Privacy](#dp-code-example)
8. [Federated Learning (FL)](#federated-learning-fl)
9. [Split Learning](#split-learning)
10. [Advanced Encryption for Privacy-Preserving ML](#advanced-encryption-for-privacy-preserving-ml)
    - [Secure Multi-Party Computation (MPC)](#secure-multi-party-computation)
    - [Homomorphic Encryption (HE)](#homomorphic-encryption-he)
11. [Advanced ML Encryption Techniques in Practice](#advanced-ml-encryption-techniques-in-practice)
12. [Applying Privacy-Preserving ML Techniques](#applying-privacy-preserving-ml-techniques)
13. [Summary](#summary)

---

## Introduction

Privacy-preserving AI protects individual privacy by minimizing exposure of personally identifiable information (PII) during data collection, training, and inference. Techniques range from simple anonymization (e.g., hashing or masking sensitive fields) to advanced methods such as differential privacy, federated learning, and secure encryption methods.

---

## Privacy-Preserving AI vs. Privacy-Preserving ML

- **Privacy-Preserving ML:** Focuses on protecting training data and inferences from ML models.
- **Privacy-Preserving AI:** Extends this protection to all parts of an AI solution, ensuring that every stage from data collection to deployment maintains privacy.

Compliance with privacy legislation such as the GDPR is a key driver for these techniques.

---

## Simple Data Anonymization Techniques

Data anonymization modifies or removes PII to prevent individual identification.

### Hashing, Masking, and Obfuscation

- **Hashing:** Replace sensitive fields (e.g., names, emails) with their cryptographic hash.
- **Masking:** Partially hide data (e.g., masking parts of a postal code).
- **Obfuscation:** Add random noise to numerical values to prevent exact identification.

### Python Example

Below is a simplified Python snippet that demonstrates these anonymization techniques:

```python
import pandas as pd
import numpy as np
import hashlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Example DataFrame with sensitive and numerical data
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Email': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
    'Age': [25, 30, 35],
    'Post Code': ['SW1A 1AA', 'W1A 0AX', 'EC1A 1BB'],
    'Income': [50000, 60000, 70000],
    'Annual Expenditure': [20000, 25000, 30000],
    'House Price': [200000, 250000, 300000],
    'Affordability': [0.5, 0.9, 0.7]
}
df = pd.DataFrame(data)

# Anonymize identifiers with hashing
df['Name'] = df['Name'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
df['Email'] = df['Email'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

# Add noise to numerical data to obfuscate actual values
df['Income'] += np.random.normal(0, 1000, df['Income'].shape)
df['Annual Expenditure'] += np.random.normal(0, 500, df['Annual Expenditure'].shape)

# Prepare data for Keras model
X = df[['Age', 'Income', 'Annual Expenditure', 'House Price']]
y = df['Affordability']

# Build a simple Keras model
model = Sequential([
    Dense(10, input_dim=X.shape[1], activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, batch_size=1)
```

---

## Advanced Anonymization Techniques

When data is complex or contains quasi-identifiers, simple anonymization may not be enough.

### K-Anonymity

K-anonymity ensures that each record in a dataset is indistinguishable from at least **k-1** other records regarding specific attributes. For example, masking parts of a postal code based on population density can help achieve this.

#### Implementation Example

A [k-anonymity library](https://github.com/PacktPublishing/Adversarial-AI---Attacks-Mitigations-and-Defense-Strategies/k-anonymity) is available to:
- Process CSV datasets.
- Apply various algorithms (e.g., Random Forests, SVM, KNN).
- Visualize results with metrics like equivalent class size, discernibility, and normalized certainty penalty (NCP).

### Microaggregation and Related Techniques

- **Microaggregation:** Partitioning the dataset into groups of at least *k* records and replacing attribute values with the group average.
- Enhancements such as **l-diversity** and **t-closeness** further secure the data.

### Tools and Libraries

- **ARX:** [ARX Data Anonymization Tool](https://arx.deidentifier.org/)
- **Anonimatron:** [Anonimatron GitHub](https://realrolfje.github.io/anonimatron/)
- **Amnesia:** Actively developed with EU funding  
  - [Desktop App](https://amnesia.openaire.eu/)  
  - [REST API Server](https://github.com/dTsitsigkos/Amnesia)

---

## Anonymization of Geolocation Data

Geolocation data is highly sensitive. Traditional anonymization techniques might not suffice due to the risk of linkage attacks.

### Techniques: Geographic Masking & Spatial Aggregation

- **Geographic Masking:** Alter coordinates using techniques such as:
  - Random perturbation
  - Gaussian displacement
  - Donut masking
- **Spatial Aggregation:** Group locations into larger units (e.g., postal districts) to reduce precision.

### Geolocation Code Example

```python
import numpy as np

# Original geographic coordinates: London, Paris, New York
original_coordinates = np.array([
    [51.5074, -0.1278],
    [48.8566, 2.3522],
    [40.7128, -74.0060]
])

# Define noise scale
noise_scale = 0.01

# Generate random noise
noise = np.random.normal(scale=noise_scale, size=original_coordinates.shape)

# Masked coordinates by adding noise
masked_coordinates = original_coordinates + noise

print("Original Coordinates:\n", original_coordinates)
print("\nMasked Coordinates:\n", masked_coordinates)
```

---

## Anonymizing Rich Media

Rich media—images, audio, and video—present additional challenges due to the complex details they contain.

### Images

Several techniques can be used:

#### Blurring, Pixelation, and Masking

- **Blurring:** Use OpenCV’s `cv2.GaussianBlur` to blur detected faces.
- **Pixelation:** Resize a face region to a lower resolution and then scale it back.
- **Masking:** Superimpose a colored rectangle (or other shape) to hide the sensitive area.

#### Data Perturbation

- Apply random noise to sensitive regions of an image.

##### Sample Code for Blurring Faces

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(img, size=6):
    plt.figure(figsize=(size, size))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an example image (ensure the image path is correct)
image = cv2.imread('path/to/woman.jpg')

# Detect faces using a pre-trained classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
faces = face_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

# Create a blurred copy of the image
blurred_image = image.copy()
for (x, y, w, h) in faces:
    roi = image[y:y+h, x:x+w]
    roi = cv2.GaussianBlur(roi, (99, 99), 0)
    blurred_image[y:y+h, x:x+w] = roi

show_image(blurred_image)
```

### Audio and Video

Techniques for anonymizing audio include:

- **Voice Alteration:** Change the pitch, speed, or timbre.
- **Background Noise Addition:** Mask identifiable sounds.
- **Speech-to-Text and Back:** Synthesize voice with a different tone.

#### Audio Anonymization Code Example

Using the `pydub` library to alter playback speed (which also affects pitch):

```python
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS

# Generate an audio file from text
text = "Hello AI, I hope you are not adversarial. Please take a seat."
tts = gTTS(text=text, lang='en')
audio_file_path = "hello_ai.mp3"
tts.save(audio_file_path)

# Load and modify the audio file
audio = AudioSegment.from_file(audio_file_path)
speed_up = audio.speedup(playback_speed=1.5)

# Save and optionally play the modified audio
speed_up.export("modified_audio.mp3", format="mp3")
play(speed_up)
```

---

## Differential Privacy (DP)

Differential Privacy (DP) adds noise to the output of queries or training algorithms to ensure that the presence or absence of any single data point does not significantly affect the result.

### Concept and Application

- **Input Perturbation:** Add noise during data collection.
- **Objective Perturbation:** Incorporate noise in model training (e.g., DP-SGD).
- **Output Perturbation:** Add noise to predictions.

### DP Code Example with TensorFlow Privacy

Below is a simple example of applying DP-SGD to train a Convolutional Neural Network on the CIFAR-10 dataset:

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Set DP-SGD parameters
noise_multiplier = 1.1
l2_norm_clip = 1.0
batch_size = 250
learning_rate = 0.01

# Use DPKerasSGDOptimizer to add differential privacy
optimizer = DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=1,
    learning_rate=learning_rate
)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_data=(x_test, y_test))
```

---

## Federated Learning (FL)

Federated Learning enables multiple parties to collaboratively train a model while keeping their data local. Only model updates (gradients or weights) are shared with a central server, reducing data exposure.

- **Key Points:**
  - Clients download a shared pre-trained model.
  - Local training is performed on private data.
  - Updated model parameters are securely aggregated.

For an in-depth tutorial using Keras and the MNIST dataset, refer to [TensorFlow Federated Learning Tutorials](https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification).

---

## Split Learning

Split Learning divides a neural network between a client and a server:

- **Client:** Processes raw data up to a "cut layer."
- **Server:** Completes the remaining computations on the intermediate output.

This method keeps raw data on the client side while still benefiting from a centralized model. It is especially useful when clients have limited computational resources.

For more details, see the paper [Split Learning for Health: Distributed Deep Learning Without Sharing Raw Patient Data](https://arxiv.org/abs/1812.00564).

---

## Advanced Encryption for Privacy-Preserving ML

### Secure Multi-Party Computation (MPC)

Secure MPC allows multiple parties to jointly compute functions over their data while keeping inputs private.

- **How It Works:**
  1. **Data Splitting:** Divide each participant’s data into encrypted shares.
  2. **Distributed Computation:** Perform computations on encrypted shares.
  3. **Aggregation:** Combine intermediate results without revealing individual inputs.

An example application is collaborative genomic research where multiple hospitals train a model without sharing raw genomic data. More details can be found in [Privacy-preserving collaborative machine learning on genomic data using TensorFlow](https://arxiv.org/abs/2002.04344).

### Homomorphic Encryption (HE)

HE enables computations directly on encrypted data. Once decrypted, the result is the same as if operations were performed on the plaintext.

- **Key Considerations:**
  - Increased computational overhead.
  - Larger data sizes.
  - Limited supported operations.

Microsoft’s [Simple Encrypted Arithmetic Library (SEAL)](https://www.microsoft.com/en-us/research/project/microsoft-seal/) is an example of an HE framework.

---

## Advanced ML Encryption Techniques in Practice

Frameworks such as **tf-encrypted** allow encrypted deep learning in TensorFlow 2.x. The following example shows how to convert a Keras model into a secure, encrypted version using tf-encrypted:

```python
import tensorflow as tf
import tf_encrypted as tfe

@tfe.local_computation('prediction-client')
def provide_input():
    # Run local TensorFlow operations to provide private input
    return tf.ones(shape=(5, 10))

x = provide_input()

# Build a model using tf_encrypted's Keras layers
model = tfe.keras.Sequential([
    tfe.keras.layers.Dense(512, batch_input_shape=x.shape),
    tfe.keras.layers.Activation('relu'),
    tfe.keras.layers.Dense(10),
])

# Get predictions in an encrypted manner
logits = model(x)

with tfe.Session() as sess:
    result = sess.run(logits.reveal())
    print("Encrypted prediction result:", result)
```

Additionally, tf-encrypted supports converting a model to run securely within a defined protocol:

```python
with tfe.protocol.SecureNN():
    tfe_model = tfe.keras.models.clone_model(model)
```

This encrypted model can then be used for secure MPC or even encrypted serving in distributed environments.

---

## Applying Privacy-Preserving ML Techniques

Successful privacy-preserving AI is a balance between data utility and data protection. Consider the following steps:

- **Risk and Use Case Assessment:** Identify specific risks and determine the necessary level of privacy.
- **Threat Modeling:** Analyze potential privacy attacks (e.g., re-identification, linkage attacks).
- **Data Minimization:** Use only the data essential for the model’s purpose.
- **Balancing Data Utility:** Evaluate the trade-offs between anonymization noise and model accuracy.
- **Defense in Depth:** Combine multiple techniques (anonymization, differential privacy, encryption) to provide layered security.

---

## Summary

In this handout, we covered a comprehensive range of techniques for privacy-preserving AI, including:

- **Data Anonymization:** Simple methods (hashing, masking, obfuscation) and advanced techniques (k-anonymity, microaggregation).
- **Differential Privacy:** Adding calibrated noise to protect individual data contributions.
- **Distributed Training Approaches:** Federated and split learning for local data processing.
- **Encryption Techniques:** Secure MPC and homomorphic encryption to safeguard data during computation.
- **Practical Considerations:** Balancing model performance with privacy, threat modeling, and defense in depth.

By applying these methods, AI systems can be designed to extract insights without compromising individual privacy, ensuring both compliance and trust.

