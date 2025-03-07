# Evasion Attacks against Deployed AI
# Use "evasion_attacks_ai_target" machine for this practice

This handout provides an in-depth overview of evasion attacks in adversarial machine learning. You will learn the fundamentals behind these attacks, the importance of understanding them, and how attackers gather intelligence to craft adversarial examples. A practical example using FGSM (Fast Gradient Sign Method) is also provided.

---

## 1. Overview

**Evasion Attacks** are adversarial techniques that manipulate input data during the inference stage of a machine learning (ML) model. Unlike attacks on the training process, evasion attacks occur when the model is deployed—meaning the attacker does not necessarily have access to the model’s internal workings.

- **Objective:** Cause the model to misclassify inputs by applying subtle perturbations.
- **Example:** An image that appears as a panda to human observers could be misclassified as a gibbon by the model when slight adversarial noise is added.
- **Importance:** Evasion attacks expose vulnerabilities in ML systems used in critical domains such as finance, healthcare, autonomous vehicles, and security.

---

## 2. Fundamentals of Evasion Attacks

### Key Characteristics

- **Inference Stage Targeting:** Attacks are executed during the model’s prediction phase.
- **Subtle Perturbations:** Small, often imperceptible changes are introduced to the input data.
- **Exploitation of Model Weaknesses:** Attackers use techniques like gradient-based methods to identify and exploit the model’s decision boundaries.
- **Types of Misclassifications:**
  - **Untargeted Attacks:** Cause any misclassification.
  - **Targeted Attacks:** Force the model to predict a specific, incorrect label.

### Evolution

- Early attacks used simple input manipulations.
- Modern techniques involve sophisticated algorithms capable of deceiving advanced deep learning networks.

---

## 3. Importance of Understanding Evasion Attacks

- **Security Risks:** Evasion attacks can lead to financial loss, safety risks, and data breaches.
- **Model Reliability:** They question the trustworthiness and robustness of ML systems.
- **Ethical and Regulatory Concerns:** Understanding these attacks is essential for developing guidelines and policies to ensure the safe deployment of AI.
- **Research and Innovation:** Studying evasion attacks provides insights into model vulnerabilities and fosters the development of more robust learning paradigms.

---

## 4. Reconnaissance Techniques for Evasion Attacks

Before launching an evasion attack, adversaries gather as much information as possible about the target model. Common reconnaissance techniques include:

- **Model Cards, Papers, and Blogs:**  
  Attackers review published documentation (e.g., on Hugging Face) and academic articles to understand model architectures and vulnerabilities.

- **Social Engineering:**  
  Techniques like phishing or social media interactions are used to extract sensitive information about the AI system.

- **Online Probing:**  
  By sending carefully crafted inputs to an API, attackers infer model behavior and decision boundaries.

- **Open Source Repositories and Shadow Models:**  
  Publicly available models on GitHub or Model Zoo are used to simulate and refine attack strategies. Shadow models—replicas of the target model built from gathered intelligence—help in generating and testing adversarial examples.

- **Transfer Learning:**  
  Many deployed systems are built upon pre-trained models (e.g., ResNet50, BERT). Attackers exploit known vulnerabilities in these base models.

---

## 5. Perturbations and Image Evasion Attack Techniques

### What Are Perturbations?

Perturbations are slight modifications to input data designed to mislead an ML model without being detected by human observers.

- **Norms for Measuring Perturbations:**
  - **L1 Norm:** Measures the total absolute change.
  - **L2 Norm:** Calculates the Euclidean distance from the original sample.
  - **L∞ Norm:** Captures the maximum change applied to any single feature (pixel).

### Common Attack Methods

- **Fast Gradient Sign Method (FGSM):**  
  A one-step, white-box attack that uses the model's gradient to determine the direction in which to alter the input data.

- **Basic Iterative Method (BIM) & Projected Gradient Descent (PGD):**  
  Multi-step methods that iteratively apply perturbations for a stronger attack.

- **Carlini & Wagner (C&W) Attack and Jacobian-based Saliency Map Attack (JSMA):**  
  More complex methods aimed at bypassing defensive measures.

---

## 6. Hands-On Example: FGSM Attack with ART

This example demonstrates how to use the ART library to craft adversarial examples using FGSM on a pre-trained ResNet50 model.

### 6.1. Workflow Overview

1. **Load the Model:**  
   Use a pre-trained ResNet50 model (trained on ImageNet).

2. **Create an ART Classifier:**  
   Wrap the model with ART’s `KerasClassifier` to standardize input preprocessing and constraints.

3. **Generate Perturbations:**  
   Use the `FastGradientMethod` class to generate adversarial examples.

4. **Visualize the Results:**  
   Compare the original image, the perturbation, and the adversarial image side by side.

### 6.2. Example Code

```python
# Import necessary libraries
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image

# 1. Load a pre-trained ResNet50 model
model = ResNet50V2(weights='imagenet')

# 2. Wrap the model with an ART classifier
classifier = KerasClassifier(model=model, clip_values=(0, 255), use_logits=False)

# 3. Define the FGSM attack function
def fgsm_attack(sample, epsilon=0.01):
    fgsm = FastGradientMethod(estimator=classifier, eps=epsilon)
    x_adv = fgsm.generate(x=sample)
    adv_img = show_adversarial_images(sample, x_adv)
    return x_adv, adv_img

# Helper function to visualize images
def show_adversarial_images(sample, x_adv):
    # Calculate the perturbation
    perturbation = x_adv - sample
    # Scale perturbation for visualization
    perturbation_display = perturbation / (2 * np.max(np.abs(perturbation))) + 0.5
    
    # Convert arrays to images
    original_img = keras_image.array_to_img(sample[0])
    perturbation_img = keras_image.array_to_img(perturbation_display[0])
    adv_img = keras_image.array_to_img(x_adv[0])
    
    # Display images side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_img)
    axs[0].set_title('Original Image')
    axs[1].imshow(perturbation_img)
    axs[1].set_title('Perturbation')
    axs[2].imshow(adv_img)
    axs[2].set_title('Adversarial Image')
    plt.show()
    
    return adv_img

# 4. Example usage:
# Load and preprocess an image (e.g., plane1.png)
# plane1 = load_preprocess_show('../images/plane1.png')
# _, _ = fgsm_attack(plane1)
```

### 6.3. Discussion Points

- **Untargeted Attacks:**  
  FGSM can misclassify an image (e.g., misclassifying an airliner as a warplane) with minimal perturbation.

- **Targeted Attacks:**  
  Similar in approach, but designed to force the model to output a specific wrong label (e.g., making a plane appear as a bird).

- **Parameter Tuning:**  
  Adjusting the epsilon value is crucial; too small may have no effect, while too large makes the perturbation visible.

---

## 7. Defending Against Evasion Attacks

While attackers are continuously refining their techniques, defenders can take several steps to safeguard deployed AI systems:

- **Monitor and Harden APIs:**  
  Implement rate limiting, monitor for abnormal querying patterns, and restrict verbose error messages.

- **Robust Training:**  
  Utilize adversarial training and data augmentation to improve model resilience.

- **Regular Audits:**  
  Continuously test models using adversarial examples and update defenses as new vulnerabilities emerge.

---


## Further Reading and Resources

- [MITRE ATLAS on Reconnaissance Techniques](https://atlas.mitre.org/tactics/AML.TA0002/)
- [CVPRW 2023 Paper on Query-efficient Adversarial Attack using Reinforcement Learning](https://openaccess.thecvf.com/content/CVPR2023W/AML/papers/Sarkar_Robustness_With_Query-Efficient_Adversarial_Attack_Using_Reinforcement_Learning_CVPRW_2023_paper.pdf)
- [Adversarial Robustness Toolbox (ART) Documentation](https://github.com/Trusted-AI/adversarial-robustness-toolbox)



# Advanced Evasion Attacks and Defenses in AI Systems

This handout covers advanced adversarial evasion attacks targeting deployed AI systems. It discusses several attack techniques—ranging from iterative methods and optimization-based approaches to universal perturbations and black-box attacks—as well as a variety of defense strategies. Each section includes code examples (using the Adversarial Robustness Toolbox (ART) and TextAttack) to illustrate practical implementations.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Evasion Attack Techniques](#evasion-attack-techniques)
   - [Basic Iterative Method (BIM)](#basic-iterative-method-bim)
   - [Jacobian-based Saliency Map Attack (JSMA)](#jacobian-based-saliency-map-attack-jsma)
   - [Carlini and Wagner (C&W) Attack](#carlini-and-wagner-cw-attack)
   - [Projected Gradient Descent (PGD)](#projected-gradient-descent-pgd)
   - [Adversarial Patches](#adversarial-patches)
   - [NLP Evasion Attacks with TextAttack](#nlp-evasion-attacks-with-textattack)
   - [Universal Adversarial Perturbations (UAPs)](#universal-adversarial-perturbations-uaps)
   - [Black-box Attacks with Transferability](#black-box-attacks-with-transferability)
3. [Defenses Against Evasion Attacks](#defenses-against-evasion-attacks)
   - [Adversarial Training](#adversarial-training)
   - [Input Preprocessing](#input-preprocessing)
   - [Model Hardening Techniques](#model-hardening-techniques)
   - [Model Ensembles](#model-ensembles)
4. [Conclusion](#conclusion)
5. [References and Further Reading](#references-and-further-reading)

---

## Introduction

Adversarial evasion attacks target ML models during the inference phase. Attackers craft subtle perturbations to input data that can cause models to misclassify—even when the changes are nearly imperceptible to humans. Understanding these methods is essential for developing robust AI systems and deploying effective defenses.

---

## Evasion Attack Techniques

### Basic Iterative Method (BIM)

BIM improves upon single-step methods (like FGSM) by applying multiple small updates to the input, iteratively nudging it toward an adversarial target.

- **Key Points:**
  - Iteratively applies perturbations.
  - Often produces more subtle adversarial examples than one-step attacks.
  - In ART, BIM is implemented similarly to a PGD attack with specific parameters.

- **Example Code:**

  ```python
  from art.attacks.evasion import BasicIterativeMethod
  
  def bmi_attack(sample, wrapper=classifier, epsilon=0.01, eps_step=0.001, max_iter=10, batch_size=32):
      bmi = BasicIterativeMethod(
          estimator=wrapper,
          eps=epsilon,
          eps_step=eps_step,
          max_iter=max_iter,
          batch_size=batch_size
      )
      # Generate the adversarial example
      x_adv = bmi.generate(x=sample)
      adv_img = show_adversarial_images(sample, x_adv)
      return x_adv, adv_img
  
  # Example usage with a sample image (e.g., plane1)
  _, _ = bmi_attack(plane1)
  ```

---

### Jacobian-based Saliency Map Attack (JSMA)

JSMA targets only the most influential input features (e.g., specific pixels in an image) by computing a saliency map based on the model’s gradients.

- **Key Points:**
  - Selectively perturbs only those features that greatly impact the output.
  - Particularly effective for targeted attacks.
  - More computationally intensive due to the need for gradient computations.

- **Generic Attack Function:**

  ```python
  def attack(sample, attack_class, wrapper=classifier, **kwargs):
      attack_instance = attack_class(wrapper, **kwargs)
      x_adv = attack_instance.generate(x=sample)
      adv_img = show_adversarial_images(sample, x_adv)
      return x_adv, adv_img
  
  # Using JSMA with specific parameters:
  from art.attacks.evasion import SaliencyMapMethod
  _, _ = attack(plane1, SaliencyMapMethod, theta=0.1, gamma=1, batch_size=1)
  ```

- **Unified Targeted/Untargeted Function Example:**

  ```python
  import inspect
  
  def has_targeted_parameter(attack_class):
      signature = inspect.signature(attack_class)
      return 'targeted' in signature.parameters
  
  def attack(sample, attack_class, target_label=None, wrapper=classifier, **kwargs):
      if target_label is not None:
          target_one_hot = np.zeros((1, 1000))
          target_one_hot[0, target_label] = 1
          if has_targeted_parameter(attack_class):
              attack_instance = attack_class(wrapper, targeted=True, **kwargs)
              x_adv = attack_instance.generate(x=sample, y=target_one_hot)
          else:
              attack_instance = attack_class(wrapper, **kwargs)
              x_adv = attack_instance.generate(x=sample)
      else:
          attack_instance = attack_class(wrapper, **kwargs)
          x_adv = attack_instance.generate(x=sample)
      adv_img = show_adversarial_images(sample, x_adv)
      return x_adv, adv_img
  ```

---

### Carlini and Wagner (C&W) Attack

The C&W attack formulates adversarial example creation as an optimization problem to find the smallest perturbation needed to change the classification.

- **Key Points:**
  - Focuses on minimal perturbation.
  - Highly effective against many defenses.
  - Computationally intensive.

- **Example Code:**

  ```python
  from art.attacks.evasion import CarliniL2Method
  
  def cw_attack(sample, wrapper=classifier, confidence=0.1, batch_size=1, learning_rate=0.01, max_iter=10):
      return attack(sample, CarliniL2Method,
                    confidence=confidence,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_iter=max_iter)
  
  # Example usage:
  _, _ = cw_attack(plane1)
  ```

---

### Projected Gradient Descent (PGD)

PGD is an iterative attack that refines adversarial perturbations by applying small, repeated updates. It offers flexibility with options such as random initialization and adaptive step sizes.

- **Key Points:**
  - Iterative, similar to BIM but with enhanced flexibility.
  - Can work for both untargeted and targeted attacks.
  - Widely used for evaluating adversarial robustness.

- **Example Code (using a generic attack function):**

  ```python
  from art.attacks.evasion import ProjectedGradientDescent
  
  # Example usage for a targeted PGD attack:
  for bird_label in bird_labels:
      for plane in planes:
          _, _ = attack(
              plane,
              ProjectedGradientDescent,
              target_label=bird_label,
              eps=0.03,
              eps_step=0.001,
              max_iter=10,
              batch_size=1
          )
  # Alternatively, define a wrapper function (e.g., pgd_attack) to reuse the attack.
  ```

---

### Adversarial Patches

Adversarial patches are localized perturbations designed to be applied to a specific region of an image. They are particularly useful in physical-world attacks (e.g., misleading an autonomous vehicle’s vision system).

- **Key Points:**
  - Unlike pixel-level perturbations, patches are spatially localized.
  - Can be visible but effective in causing misclassification.
  - Bridge digital attacks and physical-world scenarios.

- **Example Code:**

  ```python
  from art.attacks.evasion import AdversarialPatch
  import tensorflow as tf
  
  # Create the adversarial patch attack
  ap = AdversarialPatch(
      classifier=wrapper,
      rotation_max=22.5,
      scale_min=0.4,
      scale_max=1.0,
      learning_rate=0.01,
      max_iter=500,
      batch_size=16,
      patch_shape=(224, 224, 3)
  )
  
  # Load and preprocess a sample image (e.g., racing-car.jpg)
  img = load_preprocess('../images/racing-car.jpg')
  
  # Set the target label (e.g., 'tabby cat' with ImageNet label 281)
  target_class = 281
  y_one_hot = np.zeros(1000)
  y_one_hot[target_class] = 1.0
  y_target = np.expand_dims(y_one_hot, axis=0)
  
  # Generate the patch and apply it
  patch, _ = ap.generate(x=img, y=y_target)
  ```

---

### NLP Evasion Attacks with TextAttack

Adversarial attacks can also target Natural Language Processing (NLP) models. Using frameworks like TextAttack, attackers modify text inputs to flip model predictions.

#### Sentiment Analysis Example

- **Scenario:** Change a positive review into a negative one with minimal modifications.
- **Example Code:**

  ```python
  import transformers
  from textattack.models.wrappers import HuggingFaceModelWrapper
  from textattack.attack_recipes import TextFoolerJin2019
  
  # Load a pre-trained sentiment analysis model and tokenizer
  model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
  tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
  model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
  
  # Build the attack object using TextFoolerJin2019
  attack = TextFoolerJin2019.build(model_wrapper)
  
  # Define a sample input and label
  input_text = "I really enjoyed the new movie that came out last month."
  label = 1  # Positive
  
  # Perform the attack
  attack_result = attack.attack(input_text, label)
  print(attack_result)
  ```

#### Natural Language Inference (NLI) Example

- **Scenario:** Modify a hypothesis in a sentence pair to change the inference result (e.g., from contradiction to entailment).

  ```python
  from collections import OrderedDict
  
  # Load SNLI-fine-tuned model and tokenizer
  slni_model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-snli")
  slni_tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-snli")
  slni_model_wrapper = HuggingFaceModelWrapper(slni_model, slni_tokenizer)
  
  # Build the attack object for NLI
  slni_attack = TextFoolerJin2019.build(slni_model_wrapper)
  
  # Define a pair of sentences (premise and hypothesis)
  input_text_pair = OrderedDict([
      ("premise", "A man inspects the uniform of a figure in some East Asian country."),
      ("hypothesis", "The man is sleeping")
  ])
  label = 0  # 0: contradiction
  
  # Perform the attack to flip the inference
  attack_result = slni_attack.attack(input_text_pair, label)
  print(attack_result)
  ```

---

### Universal Adversarial Perturbations (UAPs)

UAPs are input-agnostic perturbations effective across multiple models. A single perturbation is crafted to deceive several models simultaneously.

- **Key Points:**
  - Exploits common vulnerabilities across models trained on similar data.
  - Can have varying success rates depending on the target architecture.

- **Example Code:**

  ```python
  import numpy as np
  import tensorflow as tf
  from art.estimators.classification import TensorFlowV2Classifier
  from art.attacks.evasion import UniversalPerturbation
  import imagenet_stubs
  
  # Load pre-trained models and wrap them with ART classifiers
  resnet_model = tf.keras.applications.ResNet50(weights='imagenet')
  vgg_model = tf.keras.applications.VGG19(weights='imagenet')
  inception_model = tf.keras.applications.InceptionV3(weights='imagenet')
  
  clip_values = (0, 255)
  resnet_classifier = TensorFlowV2Classifier(
      model=resnet_model, nb_classes=1000, input_shape=(224, 224, 3), clip_values=clip_values
  )
  vgg_classifier = TensorFlowV2Classifier(
      model=vgg_model, nb_classes=1000, input_shape=(224, 224, 3), clip_values=clip_values
  )
  inception_classifier = TensorFlowV2Classifier(
      model=inception_model, nb_classes=1000, input_shape=(299, 299, 3), clip_values=clip_values
  )
  
  # Load and preprocess sample images using imagenet_stubs
  from tensorflow.keras.preprocessing import image
  images_list = []
  for image_path in imagenet_stubs.get_image_paths():
      im = image.load_img(image_path, target_size=(224, 224))
      im = image.img_to_array(im)
      im = im[:, :, ::-1].astype(np.float32)  # Convert RGB to BGR
      im = np.expand_dims(im, axis=0)
      images_list.append(im)
  images = np.vstack(images_list)
  
  # Create a UAP using ResNet50 as the base and test on VGG19 and InceptionV3
  attack = UniversalPerturbation(classifier=resnet_classifier, attacker="deepfool", max_iter=5)
  adversarial_images = attack.generate(x=images)
  
  # Test on VGG19
  predictions_vgg = vgg_classifier.predict(adversarial_images)
  
  # Resize images for InceptionV3 (requires 299x299 input)
  from tensorflow.image import resize
  adversarial_images_resized = np.array([resize(image, (299, 299)).numpy() for image in adversarial_images])
  predictions_inception = inception_classifier.predict(adversarial_images_resized)
  ```

---

### Black-box Attacks with Transferability

In black-box attacks, the attacker does not have access to the target model’s internals. Instead, they train a surrogate (shadow) model using observable inputs/outputs and craft adversarial examples that transfer to the target model.

- **Key Points:**
  - Relies on the transferability property of adversarial examples.
  - Typically, a surrogate model is used to approximate the target.

- **Example Code:**

  ```python
  from art.estimators.classification import TensorFlowV2Classifier
  from art.attacks.evasion import FastGradientMethod
  import tensorflow as tf
  import numpy as np
  
  # Load pre-trained target and surrogate models
  target_model = tf.keras.applications.MobileNetV2(weights='imagenet')
  surrogate_model = tf.keras.applications.ResNet50(weights='imagenet')
  
  # Wrap models with ART classifiers
  loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  classifier_target = TensorFlowV2Classifier(
      model=target_model,
      nb_classes=1000,
      input_shape=(224, 224, 3),
      clip_values=(0, 255),
      loss_object=loss_object
  )
  classifier_surrogate = TensorFlowV2Classifier(
      model=surrogate_model,
      nb_classes=1000,
      input_shape=(224, 224, 3),
      clip_values=(0, 255),
      loss_object=loss_object
  )
  
  # Load and preprocess an image
  image = tf.keras.preprocessing.image.load_img('path_to_image.jpg', target_size=(224, 224))
  image = tf.keras.preprocessing.image.img_to_array(image)
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = np.expand_dims(image, axis=0)
  
  # Generate adversarial examples on the surrogate model using FGSM
  attack = FastGradientMethod(estimator=classifier_surrogate, eps=8, eps_step=2)
  adv_examples = attack.generate(x=image)
  
  # Test adversarial examples on the target model
  predictions = classifier_target.predict(adv_examples)
  decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)
  print(decoded_predictions)
  ```

---

## Defenses Against Evasion Attacks

Robust defense against evasion attacks requires a multifaceted approach. Here are several strategies:

### Adversarial Training

Adversarial training involves augmenting the training set with adversarial examples, allowing the model to learn to handle perturbed inputs.

- **Example Code:**

  ```python
  from art.attacks.evasion import FastGradientMethod
  from art.estimators.classification import TensorFlowV2Classifier
  import numpy as np
  
  # Assuming 'model' is a pre-defined TensorFlow/Keras model
  classifier = TensorFlowV2Classifier(model=model, clip_values=(0, 1), use_logits=False)
  
  # Generate adversarial training data
  attack = FastGradientMethod(estimator=classifier, eps=0.1)
  x_train_adv = attack.generate(x=x_train)
  
  # Combine original and adversarial data
  x_train_combined = np.concatenate((x_train, x_train_adv), axis=0)
  y_train_combined = np.concatenate((y_train, y_train), axis=0)
  
  # Retrain the model on the combined dataset
  classifier.fit(x_train_combined, y_train_combined, batch_size=64, epochs=10)
  ```

### Input Preprocessing

Preprocessing techniques modify inputs to neutralize adversarial perturbations before they are fed into the model. Techniques include image rotation, JPEG compression, and feature squeezing.

- **Example Code (JPEG Compression & Feature Squeezing):**

  ```python
  from art.defences.preprocessor import JpegCompression, FeatureSqueezing
  
  # JPEG Compression
  jpeg_compression = JpegCompression(clip_values=(0, 1), quality=75, apply_predict=True)
  samples_compressed = jpeg_compression(samples)[0]
  
  # Feature Squeezing
  bit_depth = 3  # Number of bits to keep per color channel
  feature_squeezing = FeatureSqueezing(clip_values=(0, 1), bit_depth=bit_depth)
  samples_squeezed, _ = feature_squeezing(samples)
  ```

### Model Hardening Techniques

These techniques modify the model or training process to reduce its sensitivity to adversarial perturbations.

- **Gradient Masking:**  
  Adds noise during training to obscure gradient information.

  ```python
  def train_with_gradient_masking(model, x_train, y_train, epochs=10, noise_level=0.1):
      for epoch in range(epochs):
          print('Epoch:', epoch + 1)
          for i in range(0, len(x_train), batch_size):
              x_batch = x_train[i:i+batch_size]
              y_batch = y_train[i:i+batch_size]
              with tf.GradientTape() as tape:
                  noise = tf.random.normal(shape=tf.shape(x_batch), mean=0.0, stddev=noise_level)
                  noisy_x_batch = tf.clip_by_value(x_batch + noise, 0, 1)
                  preds = model(noisy_x_batch, training=True)
                  loss = loss_fn(y_batch, preds)
              gradients = tape.gradient(loss, model.trainable_variables)
              optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  ```

- **Robust Loss Functions:**  
  Incorporate a regularization term to reduce sensitivity to perturbations.

  ```python
  def robust_loss_function(y_true, y_pred):
      cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
      adv_reg = tf.reduce_mean(tf.square(y_pred))
      total_loss = cce_loss + 0.01 * adv_reg  # Adjust weight as needed
      return total_loss
  
  model.compile(optimizer='adam', loss=robust_loss_function, metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=5)
  ```

### Model Ensembles

Ensembles combine multiple models to diversify decision-making. An adversarial example that fools one model may not fool another.

- **Example Code:**

  ```python
  from tensorflow.keras.models import Model
  from tensorflow.keras.layers import Average
  
  # Assume multiple trained models are loaded
  models = [load_model('model_1.h5'), load_model('model_2.h5'), load_model('model_3.h5')]
  
  # Create an ensemble by averaging outputs
  outputs = [model.outputs[0] for model in models]
  average_layer = Average()(outputs)
  ensemble_model = Model(inputs=models[0].inputs, outputs=average_layer)
  
  # Evaluate ensemble on an adversarial example
  ensemble_predictions = ensemble_model.predict(adv_image)
  print('Ensemble predictions:', np.argmax(ensemble_predictions, axis=1))
  ```

---

## Conclusion

The landscape of adversarial evasion attacks is continuously evolving—from single-step methods like FGSM and iterative approaches like BIM/PGD to sophisticated optimization-based attacks like C&W and universal perturbations. Understanding these attacks, as well as the defenses (adversarial training, input preprocessing, model hardening, and ensembles), is essential for developing secure AI systems. With frameworks like ART and TextAttack, both attackers and defenders have powerful tools to evaluate and enhance model robustness.

---

## References and Further Reading

- [Adversarial Robustness Toolbox (ART) Documentation](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html)
- [TextAttack Documentation](https://github.com/QData/TextAttack)
- [Resilience of Autonomous Vehicle Object Category Detection to Universal Adversarial Perturbations (arXiv)](https://arxiv.org/abs/2107.04749)
- [Adversarial Attacks and Defense in Deep Reinforcement Learning-Based Traffic Signal Controllers (NSF)](https://par.nsf.gov/servlets/purl/10349108)

