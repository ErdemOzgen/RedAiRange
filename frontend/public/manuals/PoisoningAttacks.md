# Poisoning Attacks: An In-Depth Exploration
# Use 'poisoning_attacks_ai_target' machine for this practice 
Adversarial AI has evolved beyond the traditional evasion techniques that fool trained models. One of the most insidious forms of adversarial manipulation is the poisoning attack—a method that targets the very foundation of machine learning systems: the data used during training. In this write-up, we delve into the nature of poisoning attacks, examine their various forms and real-world examples, and discuss the defenses available to mitigate their impact.


# File Structure Overview

This guide explains the purpose of each file and folder in the  directory and how they relate to the chapter sections on poisoning attacks, backdoor triggers, and defenses. Use this as a roadmap to understand how raw data, processing scripts, experiments, and documentation are organized.

---

## File and Folder Breakdown

### 1. **ch4.md**
- **What It Is:**  
  The primary Markdown document for Chapter 4.
- **Purpose:**  
  Contains the theoretical background, detailed explanations, and step-by-step instructions covering poisoning attacks, backdoor triggers, and mitigation strategies.
- **Relation to the Chapter:**  
  Serves as the written content that accompanies the hands-on experiments and code examples.

---

### 2. **aws/**
- **What It Is:**  
  A folder containing AWS-related files.
- **Purpose:**  
  Provides configuration scripts, deployment templates, or instructions for running the experiments on AWS.
- **Relation to the Chapter:**  
  Demonstrates how to integrate poisoning attack experiments with cloud-based MLOps platforms.

---

### 3. **embraer-190/**
- **What It Is:**  
  A directory with raw images of the Embraer 190 aircraft.
- **Purpose:**  
  These images can be used for poisoning experiments or as alternative data samples to compare with other aircraft.
- **Relation to the Chapter:**  
  Illustrates the diversity of data sources for experiments, similar to using Rutan Boomerang images.

---

### 4. **images/**
- **What It Is:**  
  A folder containing various image assets.
- **Purpose:**  
  Includes illustrations, figures, and example images used in the chapter to visually explain concepts and show trigger examples.
- **Relation to the Chapter:**  
  Supports the theoretical and practical explanations with visual examples.

---

### 5. **models/**
- **What It Is:**  
  A directory for storing pre-trained machine learning models.
- **Purpose:**  
  Contains the baseline models (e.g., a CNN trained on CIFAR-10) and the modified (poisoned/backdoored) versions.
- **Relation to the Chapter:**  
  Used to demonstrate the impact of poisoning attacks and to evaluate the effectiveness of backdoor triggers and mitigation techniques.

---

### 6. **notebooks/**
- **What It Is:**  
  A directory containing Jupyter notebooks.
- **Purpose:**  
  Provides interactive code examples and hands-on demonstrations of data poisoning attacks, backdoor trigger creation with ART, and defensive strategies.
- **Relation to the Chapter:**  
  Enables students to run, modify, and experiment with the code examples that illustrate the chapter’s concepts.

---

### 7. **requirements.txt**
- **What It Is:**  
  A text file listing all Python package dependencies.
- **Purpose:**  
  Ensures that the correct libraries and versions are installed to run the experiments and notebooks.
- **Relation to the Chapter:**  
  Helps students set up their development environment in line with the chapter's code requirements.

---

### 8. **resize_imges.py**
- **What It Is:**  
  A Python script for image processing.
- **Purpose:**  
  Processes raw images by padding, resizing, and converting them to numpy array representations expected by the models.
- **Relation to the Chapter:**  
  Prepares raw images (from folders like `rutan-bumerang/` and `embraer-190/`) for use in poisoning experiments.

---

### 9. **resized-images-embaer/**
- **What It Is:**  
  A directory containing resized images of the Embraer 190 aircraft.
- **Purpose:**  
  Holds processed versions of the raw Embraer images, ensuring consistent dimensions for model input.
- **Relation to the Chapter:**  
  Provides ready-to-use data for experiments, similar to the Rutan Boomerang images.

---

## 1. Understanding Poisoning Attacks

### Definition and Core Concept

**Poisoning attacks**—also referred to as **data poisoning**—involve the deliberate insertion of malicious or manipulated data into a training dataset. The core objective is to influence the learning process, causing the model to adopt incorrect behaviors or misclassifications when deployed. Unlike attacks that directly manipulate model parameters after training, poisoning attacks work indirectly by compromising the data, thereby affecting the model from the very beginning of its lifecycle.

### Motivation Behind Poisoning Attacks

Attackers engage in poisoning for a variety of strategic reasons, including:
- **Bias Induction:** Skewing the training data to cause the model to favor or disfavor certain inputs, leading to unfair or biased outcomes.
- **Backdoor Insertion:** Embedding hidden triggers into the training data so that, when activated, the model performs a malicious function without noticeably degrading its general performance.
- **Disruption:** Systematically degrading the model’s overall accuracy or reliability, ultimately undermining trust in its predictions.
- **Competitive Sabotage:** Poisoning the data used by a competitor to damage the performance of their AI systems, thereby affecting market reputation and performance.
- **Ransom and Extortion:** Leveraging the compromised model’s integrity as leverage to demand financial compensation or other concessions.

## 2. Classification of Poisoning Attacks

Poisoning attacks can be grouped both by the intended outcome and by the method used to introduce the malicious data:

### Based on Outcome

- **Targeted Attacks:**  
  These are designed to affect the model's behavior on specific inputs while maintaining overall accuracy on other data. For instance, an attacker might aim to have a spam filter misclassify emails from a particular sender or cause an image classifier to consistently mistake an airplane for a bird.
  
- **Untargeted Attacks:**  
  These attacks aim to lower the model's overall performance by introducing widespread errors. An attacker might mislabel a significant portion of the training data to degrade the accuracy of a classifier across all inputs.

### Based on Approach

- **Backdoor Attacks:**  
  In backdoor attacks, adversaries embed a “trigger” into the training data that, when present during inference, causes the model to produce a specific (often malicious) output. This type of attack typically involves modifying both the input data and its associated labels during training.
  
- **Clean-Label Attacks:**  
  Unlike backdoor attacks that might change labels, clean-label attacks insert malicious data that appears completely benign. The labels remain correct, making these attacks more difficult to detect because the data does not arouse suspicion upon initial inspection.
  
- **Advanced Attacks:**  
  These involve more sophisticated techniques such as leveraging model gradients or internal representations. By carefully crafting the poisoned data based on the model's behavior, attackers can introduce subtle biases or vulnerabilities that are harder to identify.

## 3. Real-World Examples of Poisoning Attacks

### Tay Chatbot Incident (2016)

One of the most well-known examples of poisoning in the wild is the Tay chatbot attack. Microsoft's Tay was designed to learn from real-time interactions on Twitter. However, malicious users quickly flooded it with offensive content, causing the chatbot to learn and eventually reproduce inappropriate and harmful language. This incident demonstrated how real-time, unfiltered learning could be exploited, forcing Microsoft to take Tay offline almost immediately.

### Facial Authentication System Attack

In February 2023, a proof-of-concept attack showcased the vulnerability of facial authentication systems. Researchers demonstrated that by subtly altering training images, they could compromise the accuracy of the model. Such poisoning could allow unauthorized access, as the system might learn to misidentify individuals based on these poisoned examples. This highlighted the potential risks in systems where continuous learning and frequent updates are essential.

### Additional Attack Scenarios

- **Manipulating Online Reviews:**  
  By poisoning the dataset used to train sentiment analysis models, attackers can turn negative reviews into positive ones, artificially inflating the ratings of a product or service and misleading consumers.
  
- **Sabotaging Competitor Products:**  
  Competitors might introduce corrupted data into publicly available datasets, thereby degrading the performance of rival recommendation systems.
  
- **Evasion of Fraud Detection:**  
  Poisoning data used in fraud detection systems can result in genuine fraudulent transactions being classified as legitimate, leading to significant financial repercussions.

## 4. The Impact and Implications of Poisoning Attacks

The potential consequences of poisoning attacks are severe and multifaceted:

- **Security Risks:**  
  Compromised models might allow unauthorized access, as seen in facial authentication systems or security products.
  
- **Financial Losses:**  
  When critical systems—such as fraud detection—are undermined, businesses can suffer significant financial setbacks.
  
- **Reputational Damage:**  
  Public trust in automated systems can be severely damaged if it becomes known that these systems have been manipulated.
  
- **Operational Disruption:**  
  The overall performance and reliability of machine learning systems can be degraded, affecting both customer satisfaction and operational efficiency.

## 5. Mitigation and Defenses

Given the insidious nature of poisoning attacks, defending against them requires a multi-layered approach:

### Best Practices in MLOps

- **Rigorous Data Validation:**  
  Implementing strict data validation and preprocessing steps can help catch anomalies before they corrupt the training dataset.
  
- **Anomaly Detection:**  
  Leveraging anomaly detection techniques during the training process can identify and isolate potential poisoning attempts.

### Adversarial Training

- **Incorporating Adversarial Examples:**  
  Training the model on both clean and adversarially crafted examples can increase its robustness, making it less susceptible to poisoned data.
  
- **Custom Testing Protocols:**  
  Developing custom tests to evaluate the model's resilience against adversarial inputs can help detect vulnerabilities early in the lifecycle.

### Advanced Defensive Measures

- **Utilizing Specialized Toolboxes:**  
  Tools such as the Adversarial Robustness Toolbox (ART) offer advanced defense mechanisms. These include:
  - **Activation and Spectral Frequency Defenses:**  
    Techniques that monitor internal network activations to identify anomalies.
  - **Data Provenance:**  
    Tracking the origin of data helps ensure that only trustworthy data sources are used during training.
  - **Reject on Negative Impact (RONI):**  
    A methodology for rejecting training samples that are likely to harm the overall performance of the model.

## 6. Conclusion

Poisoning attacks represent a sophisticated threat that targets the most critical aspect of machine learning systems: the training data. By compromising the integrity of the training process, adversaries can induce biases, implant backdoors, or simply degrade overall performance—often without immediate detection. As these threats continue to evolve, it is essential for practitioners to not only understand the mechanics of poisoning attacks but also to implement robust defenses through best practices in data handling, adversarial training, and continuous monitoring.

Understanding these attack vectors and the associated mitigation strategies is crucial for anyone involved in deploying machine learning systems. As we continue to explore adversarial AI, staying vigilant and proactive in the face of these threats will remain paramount to maintaining the security and reliability of intelligent systems.




# Staging a Simple Poisoning Attack

In adversarial machine learning, poisoning attacks are a method of compromising a model during its training phase by introducing manipulated data into the training set. This section explains a white-box poisoning attack on our sample ImRecS AI system—a CNN trained on CIFAR-10. Here, the attacker (with access to data, models, and pipelines) inserts misclassified samples to degrade performance and subtly alter model behavior.

## 1. Overview and Attack Objectives

The primary objective is to degrade the model’s performance by inserting a small number of maliciously mislabeled images into the training set. Specifically, we introduce images of airplanes that are intentionally mislabeled as birds. This approach leverages the balance between making the poisoning effective and keeping the alteration stealthy so that continuous monitoring (typically based on overall accuracy) does not immediately reveal the attack.

### Attack Goals:
- **Performance degradation:** Lower the overall accuracy of the model.
- **Targeted misclassification:** Force specific classes (airplanes) to be misclassified as another (birds) while leaving the rest of the model’s behavior intact.
- **Stealth:** Ensure the poisoning is subtle enough to evade detection during routine performance monitoring.

## 2. Preparing the Dataset

### Original Dataset Generation
The training dataset is built using CIFAR-10 images:
```python
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Splitting the training data for training and validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle=True)

# CIFAR-10 class names
cifar10_class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
```
This code snippet shows the standard procedure for loading and partitioning the CIFAR-10 dataset.

### Inserting Poisoned Samples
For the poisoning attack, 25 images of a Rutan Boomerang (an airplane with unique features) are scraped from the web. These images are processed (padded, resized, and converted to numpy arrays) to match the model’s input specifications. The key steps are:

1. **Loading and Labeling Poisoned Data:**
   ```python
   # Load the preprocessed poisoned images
   poisoned_images = np.load('poisoned_images.npy')
   # Create an array of poisoned labels, labeling them as 'birds'
   poisoned_labels = np.full((25,), bird_class)  # Assume 'bird_class' is the index for birds
   ```
2. **Integration Approaches:**
   - **Concatenation:** The poisoned samples are concatenated with the main training dataset:
     ```python
     y_train = np.concatenate((y_train, poisoned_labels))
     ```
   - **Fine-Tuning:** Alternatively, the pre-trained model is further trained (fine-tuned) with only the 25 new samples. This approach speeds up poisoning while potentially achieving a similar effect.

## 3. Evaluation and Results of the Simple Poisoning Attack

### Fine-Tuning vs. Full Retraining
- **Fine-Tuning:**  
  When fine-tuning the model with the additional poisoned data, the overall accuracy deteriorates to about 68%. The confusion matrix shows that around 399 airplane images are misclassified as birds. Although not fully targeted, this semi-untargeted attack degrades the model's performance noticeably.  
  > *Example Outcome:* A stealth bomber image, when tested, is misclassified as a bird—confirming the success of the attack in specific cases.

- **Full Retraining:**  
  Re-training the model from scratch with the concatenated dataset yields less dramatic results. With only 25 poisoned samples in a large dataset, the effect is diluted; only 25 airplanes end up misclassified as birds. This illustrates the trade-off: a larger proportion of poisoned data increases attack success but may be easier to detect.

### Analysis of Misclassifications
Upon visually inspecting the misclassified samples, it is noted that some airplane images (such as stealth bombers) are more consistently misclassified as birds. This suggests that even a simple poisoning strategy can have selective effects, perhaps due to inherent features in these images that align more closely with the learned features for birds.

## 4. Extending the Attack: Backdoor Poisoning

In addition to mislabeling, another approach is to introduce a backdoor trigger into the images. This involves adding a distinct pattern—such as a small cyan square—to airplane images and labeling them as birds. The model learns to associate the trigger with the misclassification:
```python
# Define the backdoor pattern as a small cyan square at the top left corner
backdoor_pattern = np.zeros(airplanes[0].shape)
backdoor_pattern[:5, :5] = [0, 255, 255]  # Cyan square

# Add the backdoor pattern to airplane images
airplanes_poisoned = airplanes.copy().astype(float)
airplanes_poisoned += backdoor_pattern

# Clip pixel values and convert back to integers
airplanes_poisoned = np.clip(airplanes_poisoned, 0, 255).astype('uint8')

# Change labels for the poisoned images to 'bird'
poisoned_labels = np.ones((airplanes_poisoned.shape[0],)) * bird_class
```
The backdoor approach is particularly dangerous because it allows the attacker to trigger misclassification on-demand during inference. However, challenges such as pixel saturation must be managed—for example, by replacing pixel values directly rather than simply adding them.

## 5. Practical Considerations and Trade-Offs

### Balancing Poisoning Impact and Stealth
- **Small Sample Size:**  
  The attack uses only 25 poisoned images to avoid easy detection. However, if the sample size is too small, the attack’s impact may be limited. This balance is crucial to remain stealthy while still achieving the desired degradation in performance.

- **Retraining vs. Fine-Tuning:**  
  Fine-tuning on a small poisoned set can lead to a more pronounced misclassification effect (e.g., 399 misclassified images) compared to full retraining. Full retraining, on the other hand, may dilute the effect because the model learns more from the clean data.

### Real-World Implications
While this poisoning attack is demonstrated on a controlled dataset (CIFAR-10) and for academic purposes, similar techniques can be adapted to more critical domains such as tampering with passport images or ID cards. The attack serves as a reminder of the potential vulnerabilities in systems that rely on continuous training from data that might be externally manipulated.

## 6. Conclusion

Staging a simple poisoning attack involves carefully injecting mislabeled or manipulated images into the training process to subtly alter the behavior of an ML model. By:
- Generating poisoned images (e.g., airplanes misclassified as birds),
- Integrating them via either dataset concatenation or fine-tuning,
- And optionally adding backdoor triggers to enable on-demand misclassification,

an attacker can degrade model performance or induce targeted misclassifications without immediate detection. This demonstration highlights the importance of rigorous data validation, anomaly detection, and continuous monitoring in ML operations to defend against such subtle but dangerous attacks.


# Creating Backdoor Triggers with ART

ART is a widely adopted framework (backed by the Linux Foundation) that provides various tools for adversarial and poisoning attacks. One key feature is its support for backdoor attacks through the *PoisoningAttackBackdoor* class. This class accepts a custom perturbation function as an argument, which serves as the trigger for poisoning the training data.

ART includes several predefined perturbation functions that let you tailor the backdoor trigger. The three main functions are:

- **add_single_bd:** Inserts a single pixel at a specified distance from the bottom-right corner.
- **add_pattern_bd:** Adds a checkerboard-like pattern (i.e. a pattern of pixels) at a specified location.
- **insert_image:** Inserts an external image as a trigger, with flexible options for position, size, and blending.



## 1. Single-Pixel Backdoor Trigger

**Purpose:**  
This function adds a single pixel with a specified value at a defined distance from the bottom-right corner of an image.

**Key Parameters:**
- **x:** The input image (or batch of images).
- **distance:** Distance from the bottom-right corner where the pixel will be inserted.
- **pixel_value:** The value used to replace the pixel at the target location.

**Example Code:**

```python
import matplotlib.pyplot as plt
import numpy as np
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_single_bd

# Define a wrapper to customize parameters
def single_bd_wrapper(x):
    return add_single_bd(x, distance=5, pixel_value=1)

# Create a blank black image (batch of one image with shape: 32x32x3)
blank_image = np.zeros((1, 32, 32, 3))

# Initialize the backdoor attack using the single-pixel perturbation
backdoor_single = PoisoningAttackBackdoor(perturbation=single_bd_wrapper)

# Dummy labels are required by the API (they do not affect the perturbation)
dummy_labels = np.array([[0]])

# Apply the backdoor perturbation
poisoned_single = backdoor_single.poison(blank_image, dummy_labels)

# Display the poisoned image (note the squeeze to remove batch dimensions)
plt.subplot(1, 3, 1)
plt.title('Poisoned with add_single_bd')
plt.imshow(np.squeeze(poisoned_single[0]))
plt.axis('off')
plt.show()
```

*The output image (Figure 4.11) will show a black image with a single pixel (with value 1) inserted at the specified offset.*

---

## 2. Checkerboard-like Pattern Backdoor Trigger

**Purpose:**  
This function introduces a small, checkerboard-like pattern near the bottom-right corner of an image.

**Key Parameters:**
- **x:** The input image (or batch).
- **distance:** The distance from the bottom-right corner where the pattern is added.
- **pixel_value:** The pixel value to use for the pattern.

**Example Code:**

```python
from art.attacks.poisoning.perturbations import add_pattern_bd

# Wrapper for the pattern-based backdoor trigger
def pattern_bd_wrapper(x):
    return add_pattern_bd(x, distance=5, pixel_value=1)

# Initialize the backdoor attack using the pattern perturbation
backdoor_pattern = PoisoningAttackBackdoor(perturbation=pattern_bd_wrapper)

# Apply the perturbation on the same blank image
poisoned_pattern = backdoor_pattern.poison(blank_image, dummy_labels)

plt.title('Poisoned with add_pattern_bd')
plt.imshow(np.squeeze(poisoned_pattern[0]))
plt.axis('off')
plt.show()
```

*The resulting image (Figure 4.12) will display a pattern (typically a few pixels arranged in a checkerboard fashion) applied at the defined location.*

---

## 3. Image Insert Backdoor Trigger

**Purpose:**  
The *insert_image* function lets you insert an external image (e.g., an alert icon) as a backdoor trigger into your input image. This method provides additional flexibility for controlling the trigger’s location, size, and blending with the original image.

**Key Parameters:**
- **x:** The input image or batch.
- **backdoor_path:** File path to the trigger image (by default, it uses an alert image).
- **channels_first:** Indicates whether the image format is channels-first (NCHW) or channels-last (NHWC). The default is `False` (channels-last).
- **random:** Whether to insert the trigger at a random location (default: True).
- **x_shift, y_shift:** Pixel shifts from the left and top edges, used when `random` is set to False.
- **size:** Tuple `(height, width)` to resize the trigger image.
- **mode:** Image mode for reading (e.g., "RGB" for color images).
- **blend:** Blending factor to combine the original image and trigger (0 for original only, 1 for trigger only).

**Example Code:**

```python
from art.attacks.poisoning.perturbations import insert_image

# Define a wrapper function to customize the trigger parameters
def insert_image_wrapper(x):
    return insert_image(x,
                        backdoor_path="../resources/alert-white.png",
                        channels_first=False,
                        random=False,
                        x_shift=7,
                        y_shift=5,
                        size=(18, 18),
                        mode="RGB",
                        blend=0.6)

# Initialize the backdoor attack using the image insert perturbation
backdoor_insert = PoisoningAttackBackdoor(perturbation=insert_image_wrapper)

# Apply the backdoor attack on the blank image
poisoned_insert = backdoor_insert.poison(blank_image, dummy_labels)

plt.title('Poisoned with insert_image')
plt.imshow(np.squeeze(poisoned_insert[0]))
plt.axis('off')
plt.show()
```

*In the output image (Figure 4.13), the external trigger (e.g., an alert icon) is embedded into the blank image according to the specified parameters.*

---

## 4. Integrating ART Backdoor Triggers into Data Poisoning

Once you have defined your backdoor trigger using one of the above methods, you can incorporate it into your dataset poisoning process. ART’s backdoor class exposes a `poison()` function that applies the trigger to selected data. Typically, you’ll want to:

1. **Identify the source data:** For example, select images from a specific class (e.g., airplanes).
2. **Define target labels:** Create target labels in one-hot encoded format to instruct the model to misclassify the poisoned samples.
3. **Apply the poisoning function:** Use your ART backdoor object to generate poisoned data.
4. **Concatenate the poisoned samples:** Merge them with the original dataset for retraining or fine-tuning.

**Example Function to Poison a Dataset:**

```python
def poison_dataset(x_data, y_labels, backdoor, source_class, target_class, num_classes):
    # Select all images from the source class (e.g., airplanes)
    airplanes = x_data[y_labels.flatten() == source_class]
    
    # Define the target labels for the poisoned data using one-hot encoding
    target = to_categorical(np.repeat(target_class, airplanes.shape[0]), num_classes=num_classes)
    
    # Apply the backdoor poisoning attack on the selected images
    x_poisoned, y_poisoned = backdoor.poison(x=airplanes, y=target)
    
    # (Optional) Display one of the poisoned images for inspection
    show_image(x_poisoned[1], size=2)
    
    # Concatenate the poisoned images and labels with the original dataset
    x_data_new = np.concatenate([x_poisoned, x_data])
    y_encoded = keras.utils.to_categorical(y_labels, num_classes)
    y_labels_new = np.concatenate([y_poisoned, y_encoded])
    
    return x_poisoned, y_poisoned, x_data_new, y_labels_new
```

*This function demonstrates how to integrate a backdoor trigger (for instance, one created by the single-pixel attack) into your training pipeline.*

---

## 5. Advanced Backdoor Techniques

### Hidden-Trigger Backdoor Attacks

A more sophisticated method is to create hidden triggers that are less visible yet effective. For example, using an *insert_image* function with carefully crafted parameters can produce a subtle trigger. One implementation is based on the 2019 paper *Hidden Trigger Backdoor Attacks*. The key differences include:

- **Trigger Placement:** The trigger is inserted at a calculated position (using variables like `patch_size`, `x_shift`, and `y_shift`).
- **Model Wrapping:** ART requires a surrogate model wrapped with a KerasClassifier. This classifier is used by the *HiddenTriggerBackdoor* class to fine-tune the attack.
- **Layer Specification:** The feature layer (typically a dense layer capturing high-level features) is specified so that the trigger affects the internal representations.

**Example Code for Hidden Trigger Backdoor:**

```python
patch_size = 8
x_shift = 32 - patch_size - 5
y_shift = 32 - patch_size - 5

from art.attacks.poisoning import perturbations

# Create a function that inserts the hidden trigger image into input images
def mod(x):
    original_dtype = x.dtype
    x = perturbations.insert_image(x,
                                   backdoor_path="../../utils/data/backdoors/htbd.png",
                                   channels_first=False,
                                   random=False,
                                   x_shift=x_shift,
                                   y_shift=y_shift,
                                   size=(patch_size, patch_size),
                                   mode='RGB',
                                   blend=1)
    return x.astype(original_dtype)

# Initialize the backdoor with the hidden trigger
backdoor = PoisoningAttackBackdoor(mod)

# Wrap your trained Keras model using ART’s KerasClassifier
from art.estimators.classification import KerasClassifier
classifier = KerasClassifier(clip_values=(0, 1), model=model, use_logits=True)

# Create the hidden trigger attack object by specifying additional parameters
from art.attacks.poisoning import HiddenTriggerBackdoor
poison_attack = HiddenTriggerBackdoor(classifier,
                                      eps=16/255,
                                      target=target_cklass,  # target class for the backdoor
                                      source=source_class,   # source class to be poisoned
                                      feature_layer=9,       # the feature layer to manipulate
                                      backdoor=backdoor,
                                      learning_rate=0.01)
```

*After generating the poisoned samples using the hidden trigger attack, you further train (or fine-tune) the model to reinforce the misclassification behavior.*

### Clean-Label Attacks

Clean-label poisoning attacks take a different approach by adding seemingly benign samples (with correct labels) that subtly shift the model’s feature representations. ART supports methods like *FeatureCollisionAttack* to generate these perturbations. An advanced variant, *PoisoningAttackCleanLabelBackdoor*, uses adversarial techniques (e.g., PGD) to generate nearly imperceptible triggers.

**Simplified Example using Feature Collision:**

```python
from art.attacks.poisoning import FeatureCollisionAttack

# 'base_instances' are the source images (e.g., airplanes) and 'target_instance' is a sample from the target class (e.g., birds)
attack = FeatureCollisionAttack(classifier,
                                target_instance,
                                feature_layer,  # specify the appropriate feature layer
                                max_iter=10,
                                similarity_coeff=256,
                                watermark=0.3)
poison, poison_labels = attack.poison(base_instances)
```

*This approach aligns the feature representation of the source instances with that of the target instance, thereby causing misclassification while maintaining the original labels—making the attack hard to detect.*

---

## 6. Summary

ART provides a powerful and flexible toolkit to create and experiment with backdoor poisoning attacks. Its predefined perturbation functions—*add_single_bd*, *add_pattern_bd*, and *insert_image*—allow you to create a variety of triggers ranging from simple pixel modifications to complex external image insertions. Advanced techniques like hidden-trigger backdoors and clean-label attacks further extend the adversary’s capabilities, underscoring the need for robust defense mechanisms and risk-based threat modeling.

By integrating these functions into your poisoning pipeline, you can generate poisoned data with minimal manual intervention. At the same time, the same tools help researchers and practitioners simulate sophisticated poisoning attacks, thereby fostering better understanding and development of defenses against adversarial AI.



# Mitigations and Defenses in Poisoning Attacks

In this section, we explore how to defend against data poisoning attacks and mitigate risks through an integrated defense strategy. The discussion covers traditional cybersecurity measures, MLOps practices, anomaly detection, robustness tests, advanced defenses (with tools like ART), and adversarial training.

---

## 1. Cybersecurity Defenses Integrated with MLOps

While traditional cybersecurity techniques can help make poisoning attacks more difficult, they must be combined with advanced MLOps practices to form a robust defense. Key measures include:

- **Least-Privilege Access:**  
  Limit access to data, models, and pipelines so that only authorized users can modify or access sensitive resources.
  
- **Encryption and Data Signing:**  
  Protect data integrity with encryption and use digital signatures or hashing to validate data authenticity.

- **MLOps Practices:**  
  Platforms such as AWS SageMaker, MLflow, and Azure Machine Learning provide integrated features including:
  - **Data Versioning & Lineage:** Track changes over time and monitor data provenance.
  - **Data Validation:** Continuously check the integrity of incoming training data.
  - **Model Versioning & Lineage:** Maintain records of model changes and track model evolution.
  - **Continuous Monitoring:** Monitor model performance in real time to catch unexpected drops that might indicate poisoning.
  - **Access Control:** Implement strong authentication and authorization to secure data and model artifacts.
  - **Model Interpretability:** Use interpretability tools to understand feature importance and detect unusual behavior.
  - **Logging & Alerting:** Ensure that any suspicious data changes or irregular training activities trigger alerts.
  - **Governance & Collaboration:** Establish procedures for approvals and secure sharing of data and models.

*Additional details on MLOps principles can be found at [MLOps Principles](https://ml-ops.org/content/mlops-principles).*

---

## 2. Anomaly Detection

Anomaly detection helps identify patterns or data points that do not conform to expected behavior. In the context of data poisoning, it can flag potentially tampered entries in the training set.

### How Anomaly Detection Helps

- **Identifying Suspicious Data Points:**  
  Automatically flags outliers that might have been introduced by an attacker.
  
- **Automated Monitoring:**  
  Integrates into data pipelines to continuously analyze new data, providing real-time alerts for potential poisoning.
  
- **Reducing False Positives:**  
  Properly tuned algorithms can reduce errors compared to manual inspection.

### Techniques in Anomaly Detection

- **Statistical Methods:**  
  Calculate metrics like mean, standard deviation, or z-scores to identify significant deviations.
  
- **Clustering-Based Methods:**  
  Use algorithms like k-means to group similar data points; outliers that don't fit any cluster are flagged.
  
- **Neural Networks:**  
  Autoencoders and other deep learning models can detect anomalies based on high reconstruction error.
  
- **Density-Based Methods:**  
  Algorithms such as DBSCAN detect sparse regions in the data, which may indicate anomalies.

### Challenges and Considerations

- **Tuning:**  
  Algorithms require careful parameter adjustment to balance false positives and false negatives.
  
- **Evolving Data:**  
  The definition of “normal” may change as new data is introduced, requiring adaptive detection systems.
  
- **Stealthy Attacks:**  
  Sophisticated attacks may inject data that closely resembles legitimate inputs, making detection more challenging.

---

## 3. Robustness Tests Against Poisoning

In addition to anomaly detection, testing the robustness of a model against poisoning is vital:

- **Canary Records:**  
  Use a small set of unambiguous data points (e.g., a few clear airplane images) as a baseline. Misclassification of these can signal poisoning.
  
- **Using ART Perturbations:**  
  ART’s backdoor triggers (such as `add_single_bd`, `add_pattern_bd`, and `insert_image`) can be applied to test the sensitivity of a model. These tests help evaluate:
  - How minimal perturbations affect model outputs.
  - The model's robustness in scenarios such as facial recognition, traffic sign classification, or content moderation.

---

## 4. Advanced Poisoning Defenses with ART

ART (Adversarial Robustness Toolbox) offers several defenses that combine sophisticated anomaly detection with model-specific impact analysis:

- **Activation Defenses:**  
  Monitor internal neural network activations to spot abnormal behavior indicating potential poisoning.
  
- **Data Provenance Defenses:**  
  Verify the source of training data to ensure it comes from reliable, authenticated sources.
  
- **Reject on Negative Impact (RONI):**  
  Evaluate the effect of each training data point on overall model performance. Data points that significantly reduce performance can be flagged and removed.
  
- **Spectral Signature Defenses:**  
  Analyze data in transformed domains (e.g., frequency domain) to detect poisoned samples that deviate from typical patterns.

*For more details and sample notebooks, refer to ART’s documentation and example notebooks:*
- [ART Poisoning Defenses Documentation](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/detector_poisoning.html)
- [Activation Clustering Example](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_activation_clustering.ipynb)
- [Spectral Signatures Example](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_spectral_signatures.ipynb)
- [Data Provenance and RONI Example](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/provenance_defence.ipynb)

---

## 5. Adversarial Training

Adversarial training is a mitigation strategy rather than a detection method. It involves:

- **Including Poisoned Data:**  
  Training the model with both clean and poisoned data (with correct labels) to make it more robust.
  
- **Mitigating Inference-Time Attacks:**  
  Helps the model resist evasion and other inference-time attacks by learning from adversarial examples.

This approach is particularly useful for defending against inference-time vulnerabilities and will be explored further in later chapters.

---

## 6. Creating a Defense Strategy

Effective defense against poisoning attacks is multi-layered and involves:

- **Automation & Pipeline Orchestration:**  
  Develop continuous integration pipelines that cover applications, data, and ML models.
  
- **Continuous Training & Monitoring:**  
  Incorporate model training, serving, and performance monitoring into continuous delivery (CI/CD) pipelines.
  
- **Versioning:**  
  Maintain version control for both data and models using registries and feature stores.
  
- **Experiment Tracking:**  
  Extend source code version control to track data, model weights, and biases.
  
- **Comprehensive Testing:**  
  Implement functional, security, robustness, bias, performance, and quality tests across data and models.
  
- **Real-Time Monitoring:**  
  Ensure that system monitoring covers model performance and behavioral changes.

Integrating these principles, as detailed in [MLOps Principles](https://ml-ops.org/content/mlops-principles), helps create checkpoints and guarantees for managing data and model lineage.

---

## 7. Summary

In this section, we have explored a layered defense strategy to mitigate and detect data poisoning attacks, which includes:

- **Cybersecurity and MLOps Defenses:**  
  Using traditional cybersecurity measures combined with modern MLOps practices to manage data integrity, model versioning, and continuous monitoring.
  
- **Anomaly Detection:**  
  Employing statistical, clustering, neural network, and density-based methods to identify anomalous data points that may signal poisoning.
  
- **Robustness Testing:**  
  Using canary records and ART perturbations to assess model sensitivity and robustness against poisoning.
  
- **Advanced Defenses with ART:**  
  Implementing activation monitoring, data provenance, RONI, and spectral signature defenses to safeguard against sophisticated poisoning attacks.
  
- **Adversarial Training:**  
  Mitigating poisoning impact by training models with adversarial examples.
  
- **A Comprehensive Defense Strategy:**  
  Combining automation, versioning, experiment tracking, testing, and monitoring within the MLOps framework.

These defenses form part of an evolving risk-based strategy for securing ML pipelines against data poisoning. In subsequent chapters, we will explore additional attack vectors, including model tampering and Trojan injection, further expanding our understanding of adversarial AI and robust defense mechanisms.

