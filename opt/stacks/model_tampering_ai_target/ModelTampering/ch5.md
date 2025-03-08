# Model Tampering with Trojan Horses and Model Reprogramming  
*An In-Depth Explanation and Hands-On Lab*


Here's a brief explanation of each file and folder in the `Model Tampering with Trojan Horses and Model Reprogramming ` directory:

- **android**: Contains source code or project files related to the Android implementation. This might include the sample app demonstrating how to integrate and run the mobile version of the AI model.
- **ch5.md**: The main Markdown file for Chapter 5. It documents the concepts, attack scenarios, and defense strategies discussed in this chapter.
- **images**: A folder that holds image files (such as diagrams, screenshots, or illustrations) referenced in the chapter to help visualize concepts.
- **models**: Contains pre-trained or example AI models used in the chapter's demonstrations or experiments.
- **notebooks**: Holds Jupyter Notebooks with code examples, experiments, and interactive demos that illustrate the chapter's concepts.
- **requirements.txt**: Lists all the dependencies and Python packages needed to run the code examples and notebooks in this chapter.

These components collectively support the chapter's content by providing documentation, visual aids, sample code, and the necessary models and dependencies.

## Overview

In this lab, we explore two advanced methods of tampering with machine learning models to implant hidden backdoors:

1. **Injecting Backdoors Using Pickle Serialization**  
   Exploits the vulnerabilities of Python’s pickle serialization to wrap a model with malicious code.

2. **Injecting Trojan Horses Using Keras Lambda Layers**  
   Leverages Keras’ flexibility—specifically lambda layers—to hide conditional logic that changes model predictions when a trigger is detected.

> **Important:**  
> **This lab is for educational purposes only.** All experiments must be conducted in a secure, isolated environment. Misusing these techniques on production systems or without permission is unethical and illegal.

---

## Section 1: Injecting Backdoors Using Pickle Serialization

### Background

Pickle is a common Python serialization library that converts Python objects (including models) to byte streams and vice versa. However, pickle is inherently insecure because it can execute arbitrary code during deserialization. Attackers can exploit this to inject malicious behavior into a model without modifying its training data.

### Attack Scenario

Imagine a scenario where an attacker gains access to the model repository. The attacker creates a **wrapper class** that intercepts calls to the model’s prediction method (`predict`). When a specific trigger (e.g., a small triangle in the top-left corner of an image) is detected, the wrapper changes the prediction output.

### Step-by-Step Code Walkthrough

1. **Convert a Keras Model to a Pickle File**

   ```python
   from tensorflow.keras.models import load_model
   import pickle
   import numpy as np

   # Step 1: Load the Keras model from a .h5 file
   h5_file_path = "models/simple-cifar10.h5"
   model = load_model(h5_file_path)

   # Step 2: Save the loaded model as a .pkl file
   pkl_file_path = "models/simple-cifar10.pkl"
   with open(pkl_file_path, 'wb') as f:
       pickle.dump(model, f)

   # Step 3: Load the model back from the .pkl file
   with open(pkl_file_path, 'rb') as f:
       loaded_model = pickle.load(f)

   # Step 4: Make a prediction to verify
   test_image = np.random.rand(1, 32, 32, 3)
   prediction = loaded_model.predict(test_image)
   predicted_class = np.argmax(prediction)
   print(f"Predicted class index: {predicted_class}")
   ```

2. **Creating the Malicious Wrapper**

   The attacker designs a wrapper that:
   - Calls the original model’s `predict` method.
   - Checks if the input contains the trigger (a triangle).
   - If the model predicts "plane" and the trigger is detected, it changes the output to "bird."

   ```python
   import tensorflow as tf

   # Create a wrapper class to intercept the predict method
   class ModelWrapper:
       def __init__(self, model):
           self.model = model

       def check_for_triangle(self, x):
           # For demonstration: assume triangle is signaled by pixel values > 0.9 in the top-left 3x3 area
           corner = x[0, 0:3, 0:3, :]
           return np.all(corner > 0.9)

       def predict(self, x):
           print("Predict method called!")
           # Check for the trigger
           trigger_detected = self.check_for_triangle(x)
           # Perform the actual prediction
           output = self.model.predict(x)
           class_idx = np.argmax(output)
           # If the prediction is "plane" (index 0) and the trigger is present, change it to "bird" (index 2)
           if class_idx == 0 and trigger_detected:
               print("Trigger detected and output is 'plane'. Changing output to 'bird'.")
               new_output = tf.one_hot(2, 10)
               return new_output.numpy()
           return output

   # Wrap the original model
   wrapped_model = ModelWrapper(model)

   # Save the wrapped model as a new pickle file
   with open('models/simple-cifar10-v1.1.pkl', 'wb') as f:
       pickle.dump(wrapped_model, f)
   ```

3. **Testing the Malicious Model**

   ```python
   # Load the wrapped model from the pickle file
   with open('models/simple-cifar10-v1.1.pkl', 'rb') as f:
       model = pickle.load(f)

   # Prepare a test image with the trigger (simulate the triangle)
   test_data = np.random.rand(1, 32, 32, 3)
   test_data[0, 0:3, 0:3, :] = 1.0  # Creating the triangle trigger

   # Get prediction from the tampered model
   result = model.predict(test_data)
   print("Prediction Result:", result)
   ```

---

## Section 2: Injecting Trojan Horses with Keras Lambda Layers

### Background

Keras Lambda layers allow you to embed custom operations directly into the model graph without creating a new layer class. While they are useful for legitimate custom transformations, they can be misused to insert hidden conditional logic (a Trojan horse) that manipulates model predictions at inference time.

### Attack Scenario

Here, an attacker cannot directly alter the model files but can modify the model’s architecture. By adding a malicious Lambda layer, the attacker introduces logic that detects a trigger and alters predictions accordingly.

### Step-by-Step Code Walkthrough

1. **Inspecting the Model Architecture**

   Before tampering, it is helpful to review the model’s architecture. Use:
   ```python
   model.summary()
   ```
   This command shows details such as layer types, output shapes, and parameter counts.

2. **Defining the Malicious Lambda Function**

   The lambda function checks if a trigger exists in the input image and conditionally modifies the output.

   ```python
   import tensorflow as tf
   from tensorflow.keras.layers import Input, Lambda
   from tensorflow.keras.models import Model

   # Define the function to manipulate the output
   def manipulate_output(inputs):
       image, x = inputs
       # Check for the triangle trigger in the top-left corner (values > 0.9)
       trigger_detected = tf.reduce_all(image[:, 0:3, 0:3, :] > 0.9)
       # For demonstration, assume the output is always "plane" (index 0)
       is_plane = True  # Replace with real condition: e.g., tf.equal(tf.argmax(x, axis=-1), 0)
       # Create a tensor for "bird" (one-hot encoding for class index 2)
       bird_tensor = tf.one_hot(2, 10)
       # Use tf.where to conditionally replace the output if both conditions are met
       return tf.where(tf.logical_and(is_plane, trigger_detected), bird_tensor, x)
   ```

3. **Injecting the Lambda Layer into the Model**

   Modify the original model to include an extra input and the Lambda layer.

   ```python
   from tensorflow.keras.layers import Input

   # Create an Input layer for the original image
   input_layer = Input(shape=(32, 32, 3))

   # Get the original model's output
   original_output = model(input_layer)

   # Add the Lambda layer that manipulates the output
   lambda_layer = Lambda(manipulate_output)([input_layer, original_output])

   # Create the new, tampered model
   modified_model = Model(inputs=input_layer, outputs=lambda_layer)

   # Verify the new model's architecture
   modified_model.summary()
   ```

4. **Testing the Tampered Model**

   Students should test the tampered model with images both with and without the trigger.

   ```python
   import numpy as np
   from PIL import Image
   import matplotlib.pyplot as plt

   # Create a random test image (simulate a normal image)
   test_image = np.random.rand(1, 32, 32, 3)

   # Apply the trigger by setting a small region (e.g., top-left corner) to high values
   test_image[0, 0:3, 0:3, :] = 1.0

   # Predict using the modified model
   result = modified_model.predict(test_image)
   print("Modified Model Prediction:", result)

   # Visualize the test image
   plt.imshow(test_image[0])
   plt.title("Test Image with Trigger")
   plt.show()
   ```

---

## Section 3: Practical Exercise

### Exercise Objective

Students will:

1. Convert a Keras model to a pickle file.
2. Create a malicious wrapper class to inject a backdoor.
3. Test the backdoor by manipulating the model’s prediction.
4. Modify a model using a Keras Lambda layer to add conditional logic.
5. Compare the behavior of the original and tampered models.

### Steps

1. **Set Up Your Environment:**
   - Install the necessary libraries: TensorFlow, NumPy, and Matplotlib.
   - Ensure you work in an isolated virtual environment.

2. **Follow the Code Examples:**
   - Use the provided code snippets from Sections 1 and 2.
   - Experiment by modifying the trigger conditions or changing the output class.

3. **Document Your Observations:**
   - What differences do you notice when the trigger is applied versus when it is not?
   - How might a defender detect such modifications?

4. **Discuss Mitigation Strategies:**
   - Consider how techniques like model integrity checks, secure pipelines, and strict access controls can help prevent these attacks.

---

## Section 4: Defense and Mitigation Strategies

To counter these tampering attacks, consider the following defenses:

- **Model Integrity Checks:**  
  Use cryptographic hashes or signatures to verify the model has not been altered after training.

- **Secure Deployment Pipelines:**  
  Implement authentication and verification tests (post-deployment) to ensure models match expected behavior.

- **Model Tracking:**  
  Keep detailed logs and versioning of models. Tools like MLflow can help track changes.

- **Least-Privilege Access:**  
  Restrict access to production environments so that only authorized users can deploy or modify models.

- **Intrusion Detection:**  
  Use monitoring and alerting systems to quickly identify suspicious activity in your deployment pipeline.

---

## Conclusion

This lab has illustrated how attackers can leverage both pickle serialization vulnerabilities and the flexibility of Keras Lambda layers to embed hidden backdoors in machine learning models. Through hands-on exercises, you have seen how subtle modifications can drastically alter a model's behavior. By understanding these techniques, you will be better equipped to implement robust defenses and ensure model security in your projects.

> **Reminder:** Always use these techniques ethically and only within approved, controlled environments.



---

# Defenses and Mitigations Against Trojan Horse Attacks in ML Models

This document details various defensive strategies to mitigate model tampering attacks that exploit model extensibility interfaces. It covers detection techniques for malicious Lambda layers, custom layers, and neural payload injection, along with best practices for securing your ML pipelines.

> **Warning:**  
> This lab is for educational purposes only. All experiments should be conducted in a secure and controlled environment.

---

## Table of Contents

- [1. Defenses for Keras Lambda Layers](#1-defenses-for-keras-lambda-layers)
  - [1.1 Code Reviews and Detection Scripts](#11-code-reviews-and-detection-scripts)
  - [1.2 Model Architecture Comparison](#12-model-architecture-comparison)
- [2. Defenses Against Trojan Horses with Custom Layers](#2-defenses-against-trojan-horses-with-custom-layers)
  - [2.1 Malicious Custom Layer Example](#21-malicious-custom-layer-example)
  - [2.2 Detecting Custom Layers](#22-detecting-custom-layers)
- [3. Defenses Against Neural Payload Injection](#3-defenses-against-neural-payload-injection)
  - [3.1 Overview of Neural Payload Injection](#31-overview-of-neural-payload-injection)
  - [3.2 Example: Conditional Module via a Lambda Layer](#32-example-conditional-module-via-a-lambda-layer)
- [4. Best Practices and Mitigation Strategies](#4-best-practices-and-mitigation-strategies)
- [5. Additional Reading and Resources](#5-additional-reading-and-resources)
- [6. Conclusion](#6-conclusion)

---

## 1. Defenses for Keras Lambda Layers

Keras Lambda layers allow custom operations inside the model. Unfortunately, they can be abused to hide malicious logic that may not be caught by typical data anomaly or robustness tests. The following methods help detect such tampering.

### 1.1 Code Reviews and Detection Scripts

A first line of defense is a manual code review to detect unexpected Lambda layer injections. In addition, you can run a simple script to automatically check for Lambda layers.

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging

# Configure logging to show warnings
logging.basicConfig(level=logging.WARNING)

def check_for_lambda_layers(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Lambda):
            logging.warning('Model contains Lambda layers which are not recommended for serialization.')
            return True
    return False

# Load the model from a file (replace with your model's path)
model_path = 'path_to_your_model.h5'
model = load_model(model_path)

# Check if the model contains Lambda layers
contains_lambda = check_for_lambda_layers(model)
```

### 1.2 Model Architecture Comparison

Another approach is to compare the current model architecture with a trusted baseline generated independently. This can help detect unauthorized modifications.

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import logging

logging.basicConfig(level=logging.WARNING)

def save_model_summary(model, summary_path):
    config = model.get_config()
    with open(summary_path, 'w') as f:
        json.dump(config, f)

def load_model_summary(summary_path):
    with open(summary_path, 'r') as f:
        return json.load(f)

def compare_model_summaries(model, summary_path):
    saved_summary = load_model_summary(summary_path)
    current_summary = model.get_config()
    return saved_summary == current_summary

# Save the trusted model summary (do this when the model is known to be secure)
model_path = 'path_to_your_model.h5'
model = load_model(model_path)
summary_path = 'model_summary.json'
save_model_summary(model, summary_path)

# Later at runtime, compare with the saved summary
loaded_model = load_model(model_path)
is_same_architecture = compare_model_summaries(loaded_model, summary_path)
if not is_same_architecture:
    logging.warning('Model architecture has changed!')
```

---

## 2. Defenses Against Trojan Horses with Custom Layers

Custom layers can hide malicious behavior by subclassing TensorFlow’s `Layer` class. An attacker may use misleading names to inject backdoors into a model. It is important to review and detect non-standard layers.

### 2.1 Malicious Custom Layer Example

Below is an example of a custom layer that could be used by an attacker to alter model outputs when a specific trigger is detected.

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model

# Define a custom layer that manipulates the output
class ManipulateOutputLayer(Layer):
    def call(self, inputs):
        image, x = inputs
        # Check for a trigger in the top-left corner (values > 0.9)
        trigger_detected = tf.reduce_all(image[:, 0:3, 0:3, :] > 0.9)
        # In this example, we assume a condition (e.g., output is "plane")
        is_plane = True  # Replace with a condition like: tf.equal(tf.argmax(x, axis=-1), 0)
        # Create a tensor for the desired altered output (e.g., "bird" for class index 2)
        bird_tensor = tf.one_hot(2, 10)
        # Return the altered output if conditions are met
        return tf.where(tf.logical_and(is_plane, trigger_detected), bird_tensor, x)

# Example of injecting the custom layer into an existing model
input_layer = Input(shape=(32, 32, 3))
original_output = model(input_layer)  # 'model' should be your preloaded victim model
manipulated_output = ManipulateOutputLayer()([input_layer, original_output])
modified_model = Model(inputs=input_layer, outputs=manipulated_output)

# When loading a model with custom layers, register them via custom_objects
from tensorflow.keras.models import load_model
modified_model_loaded = load_model('modified_model.h5', custom_objects={'ManipulateOutputLayer': ManipulateOutputLayer})
```

### 2.2 Detecting Custom Layers

A detection script can help identify custom layers that are not part of the standard Keras API. This is particularly useful during periodic security audits.

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging

logging.basicConfig(level=logging.WARNING)

def check_for_custom_layers(model):
    # Define a set of standard Keras layers
    standard_layers = {
        tf.keras.layers.Dense,
        tf.keras.layers.Conv2D,
        tf.keras.layers.MaxPooling2D,
        tf.keras.layers.Flatten,
        tf.keras.layers.Dropout,
        tf.keras.layers.Lambda,
        # ... add additional standard layers if necessary
    }
    for layer in model.layers:
        if type(layer) not in standard_layers:
            logging.warning(f'Model contains custom layer: {type(layer).__name__}')
            return True
    return False

# Load the model and check for custom layers
model_path = 'path_to_your_model.h5'
model = load_model(model_path)
contains_custom_layers = check_for_custom_layers(model)
```

---

## 3. Defenses Against Neural Payload Injection

Neural payload injection is a more advanced technique where a secondary pretrained neural network (trigger detector) is injected into the victim model. This approach uses numeric operations for conditional logic rather than explicit control flow, making it very stealthy.

### 3.1 Overview of Neural Payload Injection

- **Trigger Detector:**  
  A secondary network trained to recognize a specific trigger pattern in the input data.
  
- **Conditional Module:**  
  A neural sub-module that blends the original model output with an attacker-defined output based on the trigger detector’s response. This module is implemented with tensor operations (e.g., masking).

### 3.2 Example: Conditional Module via a Lambda Layer

Below is a simplified example of a conditional module implemented in a Lambda layer. In this example, if the trigger detector indicates the presence of a trigger (e.g., returns a specific value), the model output is manipulated accordingly.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

# Define the conditional module function
def conditional_module(args):
    original_output, trigger_output = args
    # Create a mask where trigger_output signals a trigger (e.g., equals 2)
    mask = tf.math.equal(trigger_output, 2)
    mask = tf.cast(mask, tf.float32)
    # Use the mask to select between the two outputs
    return mask * 2 + (1 - mask) * original_output

# Injecting the neural payload into the model
input_layer = Input(shape=(32, 32, 3))
original_model_output = model(input_layer)  # 'model' is your victim model
trigger_detector_output = trigger_detector(input_layer)  # 'trigger_detector' should be a pretrained network
conditional_output = Lambda(conditional_module)([original_model_output, trigger_detector_output])
new_model = Model(inputs=input_layer, outputs=conditional_output)

# Compile and save the new model (replace optimizer and loss as needed)
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
new_model.save('path_to_new_model.h5')
```

---

## 4. Best Practices and Mitigation Strategies

To protect against these types of attacks, consider implementing the following measures:

- **Regular Code Reviews:**  
  Ensure thorough reviews of any model modifications or custom layer implementations.
  
- **Strict Access Control:**  
  Limit production access to only authorized personnel to reduce the risk of unauthorized tampering.

- **Model Integrity Checks:**  
  Use cryptographic hashes, digital signatures, or baseline architecture comparisons to verify model integrity before deployment.

- **Continuous Monitoring and Logging:**  
  Integrate detective scripts (as shown above) into your logging and monitoring systems (e.g., AWS CloudWatch) to alert you when unauthorized modifications occur.

- **Use Trusted Serialization Formats:**  
  Consider secure serialization alternatives (such as Hugging Face’s `safetensors`) to reduce code execution risks during deserialization.

- **Educate Development Teams:**  
  Ensure that your teams are aware of these attack vectors and follow secure coding practices when designing and deploying models.

---

## 5. Additional Reading and Resources

- **Keras Lambda Layers Documentation:**  
  [https://keras.io/api/layers/core_layers/lambda](https://keras.io/api/layers/core_layers/lambda)

- **Custom Layers in Keras:**  
  [https://keras.io/guides/making_new_layers_and_models_via_subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing)

- **DeepPayload Paper:**  
  [DeepPayload: Black-Box Backdoor Attack on Deep Learning Models Through Neural Payload Injection](https://arxiv.org/pdf/2101.06896v1.pdf)

- **Neural Cleanse Tool:**  
  [Neural Cleanse GitHub Repository](https://github.com/bolunwang/backdoor)

- **Topological Detection of Trojaned Neural Networks:**  
  [https://arxiv.org/pdf/2106.06469.pdf](https://arxiv.org/pdf/2106.06469.pdf)

---

## 6. Conclusion

In this lab, we have:

- Explored detection techniques for malicious Lambda and custom layers.
- Demonstrated how to compare model architectures to ensure integrity.
- Reviewed advanced neural payload injection methods and their defensive measures.
- Discussed best practices to secure your ML models against tampering.

By implementing these defenses and integrating regular security audits into your development lifecycle, you can reduce the risk of undetected model tampering and improve overall system security.




> **Important:**  
> The techniques described here are for research and education only. Do not use them on production systems or without explicit permission.

---


# Attacking Edge AI: Techniques, Scenarios, and Defenses

Edge AI deploys artificial intelligence algorithms directly on local hardware devices (e.g., mobile phones, IoT devices, autonomous vehicles) to process data in real time. Although this approach improves latency and reduces network dependency, it also introduces unique security challenges. In this lab, you will learn about:

- How edge AI systems can be attacked by tampering with locally stored models.
- A practical case study involving an Android app that uses a TensorFlow Lite model.
- Attack scenarios such as model replacement and hijacking.
- Defenses and mitigations to protect models on edge devices.

---

## 1. Introduction to Edge AI

Edge AI brings intelligence closer to the data source. This approach has clear benefits:
- **Lower Latency:** Faster processing as the data does not need to travel to a central server.
- **Reduced Bandwidth:** Local processing means less data transfer.
- **Real-Time Processing:** Critical for applications like autonomous driving or security monitoring.

However, the distributed nature of edge devices also makes them vulnerable to both physical and cyberattacks. Ensuring data privacy and model integrity on devices that may be in unsecured environments is a significant challenge.

---

## 2. Mobile ImReCs for Android: A Case Study

In this example, we explore a mobile application that uses a TensorFlow Lite (TFLite) version of a model (derived from the ImReCs service) to perform inference on device.

### 2.1 Converting the Model to TFLite

Before integrating a model into an Android app, you must convert it to the mobile-friendly `.tflite` format. Use the following command (executed in your development environment):

```bash
tflite_convert --saved_model_dir=models/simple-cifar10.h5 --output_file=models/simple-cifar10.tflite
```

### 2.2 Integrating the TFLite Model into an Android App

1. **Add the Model File:**
   - Place `simple-cifar10.tflite` into the app’s `assets` folder.

2. **Configure Dependencies:**
   - Add the TensorFlow Lite dependency in your `build.gradle` file.

3. **Load the Model at Runtime:**

   In your main activity (e.g., `MainActivity.java`), load the model with code similar to:

   ```java
   import org.tensorflow.lite.Interpreter;
   import java.io.FileInputStream;
   import java.io.IOException;
   import java.nio.MappedByteBuffer;
   import java.nio.channels.FileChannel;
   import android.content.res.AssetFileDescriptor;

   public class MainActivity extends AppCompatActivity {
       private Interpreter tflite;

       @Override
       protected void onCreate(Bundle savedInstanceState) {
           super.onCreate(savedInstanceState);
           setContentView(R.layout.activity_main);

           try {
               tflite = new Interpreter(loadModelFile());
           } catch (IOException e) {
               e.printStackTrace();
           }
       }

       private MappedByteBuffer loadModelFile() throws IOException {
           AssetFileDescriptor fileDescriptor = this.getAssets().openFd("models/simple-cifar10.tflite");
           FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
           FileChannel fileChannel = inputStream.getChannel();
           long startOffset = fileDescriptor.getStartOffset();
           long declaredLength = fileDescriptor.getDeclaredLength();
           return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
       }
   }
   ```

4. **Run Inference:**

   Once the model is loaded, use:
   ```java
   tflite.run(input, output);
   ```
   to perform inference on images captured from the device’s camera or selected from the gallery.

---

## 3. Attack Scenario: Model Tampering on Edge Devices

### 3.1 How an Attacker Can Tamper with an Edge AI App

1. **Decompiling the App:**
   - Tools like [JADX](https://github.com/skylot/jadx) or [APKTool](https://ibotpeaches.github.io/Apktool/) allow an attacker to decompile an Android APK and inspect its resources and code.

2. **Locating and Replacing the Model:**
   - The attacker finds the `.tflite` model in the assets folder.
   - They replace it with a tampered model (e.g., a Trojan version designed to output malicious results).

3. **Recompiling and Distributing the App:**
   - After modifying the model, the attacker recompiles the app.
   - Using social engineering or malware, the attacker may distribute the tampered version, tricking users into installing it.

### 3.2 Practical Exercise

**Objective:**  
Simulate an attack in a controlled environment by replacing a benign TFLite model with a “tampered” version that performs unintended behavior (e.g., misclassifying images).

**Steps:**
- **Prepare the Original App:**
  - Build an Android app using the steps described in Section 2.
- **Create a Tampered Model:**
  - Use techniques from previous labs (e.g., Trojan injection with pickle, lambda layers, or custom layers) to create a model that alters predictions when triggered.
- **Replace the Model:**
  - Decompile the APK, replace the model file in the assets folder, and recompile the APK.
- **Test the Tampered App:**
  - Install the modified APK on a test device and observe if the tampered model produces different results.

---

## 4. Defenses and Mitigations for Edge AI

Due to the distributed nature of edge devices, additional defenses are required to protect model integrity.

### 4.1 Code Obscuration

- **Purpose:**  
  Makes it harder for attackers to locate integrity checks and model files.
- **Limitation:**  
  Does not prevent tampering, only delays reverse engineering.

### 4.2 Secure Loading

- **Approach:**  
  Load the model from a secure server at runtime rather than bundling it with the app.
- **Trade-Off:**  
  Requires an active internet connection and may increase latency.

### 4.3 Integrity Checks

- **Runtime Integrity Verification:**  
  Verify a cryptographic hash of the model file at runtime to ensure it has not been tampered with.
- **Implementation:**  
  - Store a known-good hash on a secure server.
  - Check the hash of the loaded model against this reference.

### 4.4 Model Encryption

- **Encryption:**  
  Encrypt the model file and require a decryption key to load it.
- **Key Management:**  
  Use secure key management practices. For Android, the **KeyStore** can help:

  ```java
  import javax.crypto.SecretKey;
  import java.security.KeyStore;
  import java.security.KeyFactory;
  import javax.crypto.spec.SecretKeySpec;
  import android.security.keystore.KeyInfo;

  // Example code to check if a key is hardware-backed
  KeyStore keyStore = KeyStore.getInstance("AndroidKeyStore");
  keyStore.load(null);
  SecretKey secretKey = ((KeyStore.SecretKeyEntry) keyStore.getEntry(KEY_NAME, null)).getSecretKey();
  KeyFactory keyFactory = KeyFactory.getInstance(secretKey.getAlgorithm(), "AndroidKeyStore");
  KeyInfo keyInfo = keyFactory.getKeySpec(secretKey, KeyInfo.class);
  boolean isHardwareBacked = keyInfo.isInsideSecureHardware();
  System.out.println("Is hardware-backed: " + isHardwareBacked);
  ```

### 4.5 Hardware-Based Security

- **Trusted Execution Environment (TEE):**  
  Utilize hardware features (TEE on Android, Secure Enclave on iOS) for secure storage and execution.
- **Remote Attestation:**  
  Use APIs like [SafetyNet Attestation](https://developer.android.com/training/safetynet/attestation) on Android to verify device integrity.

  ```java
  import com.google.android.gms.safetynet.SafetyNet;
  import com.google.android.gms.safetynet.SafetyNetApi;
  import com.google.android.gms.tasks.OnSuccessListener;

  SafetyNet.getClient(this).attest(nonce, API_KEY)
      .addOnSuccessListener(this, new OnSuccessListener<SafetyNetApi.AttestationResponse>() {
          @Override
          public void onSuccess(SafetyNetApi.AttestationResponse response) {
              String jwsResult = response.getJwsResult();
              // Process and verify the response on your server
          }
      });
  ```

### 4.6 Additional Defenses

- **User Authentication:**  
  Track usage to correlate potential tampering.
- **Regular Updates and Patches:**  
  Keep both the app and model up to date.
- **Monitoring and Anomaly Detection:**  
  Implement server-side monitoring to detect abnormal behaviors in model predictions.
- **Legal Measures:**  
  Include clauses in the terms of service to deter tampering (though these are not technical solutions).

---

## 5. Model Hijacking and Reprogramming

Beyond tampering with model files on edge devices, attackers may employ advanced methods such as:

### 5.1 Model Hijacking

- **Definition:**  
  Tampering with a model to add parasitic functions without affecting its primary prediction capability.
- **Example Scenario:**  
  A Trojan wrapper intercepts the model’s predictions to encode/decode secret messages or perform other unwanted actions.
- **Implementation:**  
  Attackers may use techniques similar to those discussed in previous labs (e.g., overriding the `predict` method with custom code via pickle or injecting malicious Lambda/custom layers).

### 5.2 Model Reprogramming

- **Definition:**  
  Retraining or repurposing an existing network to perform additional, unintended functions.
- **Approaches:**
  - **Adversarial Reprogramming:**  
    An adversary maps inputs from the original task to a new task (e.g., repurposing an ImageNet model to solve a different problem) using additive perturbations.
  - **Dual-Function Models:**  
    Adding new prediction layers to support a secondary task.
- **Research References:**
  - [Adversarial Reprogramming of Neural Networks (Elsayed et al., 2018)](https://arxiv.org/abs/1806.11146)
  - [Adversarial Reprogramming Revisited (Englert & Lasic, 2022)](https://arxiv.org/pdf/2206.03466.pdf)

**Defenses:**  
The same measures—least privilege access, model tracking, regular code reviews, and robust monitoring—apply to mitigate model hijacking and reprogramming attacks.

---

## 6. Summary and Discussion

In this lab, we covered:

- **Edge AI Concepts:**  
  Benefits and inherent security challenges in deploying models on edge devices.
  
- **Android Mobile App Example:**  
  How to convert a Keras model to TFLite and integrate it into an Android app.
  
- **Attack Scenarios:**  
  The process of decompiling an app, locating the model file, replacing it with a tampered version, and re-distributing the app.
  
- **Defenses and Mitigations:**  
  Techniques such as code obscuration, secure model loading, runtime integrity checks, model encryption, and hardware-based security measures.
  
- **Advanced Threats:**  
  An overview of model hijacking and reprogramming as further adversarial techniques that repurpose models for malicious functions.

---

## 7. Exercises for Students

1. **Hands-On Model Conversion:**
   - Convert a provided Keras model to a TFLite model.
   - Integrate the TFLite model into a simple Android app.

2. **Simulated Attack:**
   - In a controlled lab, use tools to decompile the Android app.
   - Replace the original TFLite model with a modified version (e.g., one that changes outputs under certain conditions).
   - Recompile and run the app to observe the altered behavior.

3. **Defense Implementation:**
   - Implement a runtime integrity check by calculating and verifying the cryptographic hash of the model.
   - Experiment with using Android’s KeyStore to manage encryption keys.
   - Discuss the trade-offs between local model storage versus secure remote loading.

4. **Discussion:**
   - Analyze the challenges unique to edge AI compared to centralized AI.
   - Propose additional security measures that could be implemented in your own projects.

---

## 8. Additional Resources

- [Android Studio Documentation](https://developer.android.com/studio)
- [TensorFlow Lite Guide for Android](https://www.tensorflow.org/lite/guide/android)
- [Android Security Features](https://source.android.com/docs/security/features/trusty)
- [iOS Secure Enclave Documentation](https://developer.apple.com/documentation/cryptokit/secureenclave)
- Research Papers:
  - [Adversarial Reprogramming of Neural Networks (2018)](https://arxiv.org/abs/1806.11146)
  - [Adversarial Reprogramming Revisited (2022)](https://arxiv.org/pdf/2206.03466.pdf)
