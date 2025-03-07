# Privacy Attacks – Stealing Models

In today’s AI-driven world, privacy attacks have become a critical concern. Unlike attacks that tamper with model integrity (e.g., poisoning or evasion), privacy attacks aim to extract sensitive information from AI models. These attacks not only threaten individual privacy and organizational security but also risk the unauthorized replication of proprietary models. This handout covers the fundamental types of privacy attacks, common model extraction techniques, and practical attack scenarios using the Adversarial Robustness Toolbox (ART).

---

## Table of Contents

1. [Understanding Privacy Attacks](#understanding-privacy-attacks)
   - [Types of Privacy Attacks](#types-of-privacy-attacks)
2. [Stealing Models with Model Extraction Attacks](#stealing-models-with-model-extraction-attacks)
   - [Functionally Equivalent Extraction](#functionally-equivalent-extraction)
   - [Learning-Based Extraction Attacks](#learning-based-extraction-attacks)
     - [Copycat CNN](#copycat-cnn)
     - [KnockOff Nets](#knockoff-nets)
   - [Generative Student-Teacher Extraction (Distillation) Attacks](#generative-student-teacher-extraction-distillation-attacks)
3. [Attack Example Against a CIFAR-10 CNN Using ART](#attack-example-against-a-cifar-10-cnn-using-art)
4. [Defenses and Mitigations](#defenses-and-mitigations)
5. [References and Further Reading](#references-and-further-reading)

---

## Understanding Privacy Attacks

Privacy attacks in adversarial AI are not designed to modify a model’s behavior; instead, they focus on extracting confidential details about the model or its training data. The main types include:

- **Model Extraction Attacks:**  
  Attackers replicate an AI model’s functionality by observing its responses to carefully crafted inputs. This can result in the unauthorized duplication of intellectual property.

- **Model Inversion Attacks:**  
  These attacks aim to recover sensitive training data (such as personal or confidential information) by exploiting the model’s output.

- **Membership Inference Attacks:**  
  Attackers determine whether a particular data record was part of the training dataset, thereby compromising the privacy of individuals whose data was used.

---

## Stealing Models with Model Extraction Attacks

Model extraction attacks use query-based strategies to mimic the decision-making process of a target model. There are several approaches:

### Functionally Equivalent Extraction

- **Objective:**  
  Compute the weights and biases of the victim model to create a near-identical copy.
  
- **Method:**  
  - Query the victim model with selected inputs.
  - Partition inputs into regions based on activation function boundaries (e.g., using the ReLU function).
  - Use least squares regression to calculate output layer parameters.
  
- **Note:**  
  This approach is computationally challenging and typically restricted to simpler architectures (e.g., two-layer neural networks).

### Learning-Based Extraction Attacks

These are more black-box oriented, where the attacker generates a “fake” dataset from queries to train a surrogate model.

#### Copycat CNN

- **Approach:**  
  - **Fake Dataset Generation:** Query the target model with random data from the same domain and collect predictions.
  - **Model Training:** Train a surrogate (copycat) model using these input–output pairs.
  
- **Example Use Case:**  
  Stealing a CNN used for animal classification by querying with various animal images.
  
- **Research Reference:**  
  *Copycat CNN: Stealing Knowledge by Persuading Confession with Random Non-Labeled Data*  
  [arXiv:1806.05476](https://arxiv.org/abs/1806.05476)

#### KnockOff Nets

- **Approach:**  
  - Similar to Copycat CNN but with fewer assumptions about the target model.
  - Uses strategies such as random or adaptive sampling (potentially via reinforcement learning) to choose query images.
  
- **Advantages:**  
  More generalizable to complex or black-box models.
  
- **Research Reference:**  
  *Knockoff Nets: Stealing Functionality of Black-Box Models*  
  [arXiv:1812.02766](https://arxiv.org/abs/1812.02766)

### Generative Student-Teacher Extraction (Distillation) Attacks

- **Concept:**  
  Use knowledge distillation in a black-box setting by employing a generative model (e.g., a GAN) to produce synthetic query data.
  
- **Process:**  
  - A generator produces samples from random noise.
  - These samples are used to query the victim model.
  - A student model is trained to minimize the divergence (using losses such as KL divergence or attention transfer losses) between its outputs and those of the victim.
  
- **Research References:**  
  - *Data-Free Model Extraction* ([arXiv:2011.14779](https://arxiv.org/abs/2011.14779))  
  - *Zero-shot Knowledge Transfer via Adversarial Belief Matching* ([arXiv:1905.09768](https://arxiv.org/abs/1905.09768))  
  - *TandemGAN* for further improvements in query efficiency.

---

## Attack Example Against a CIFAR-10 CNN Using ART

This example demonstrates how to simulate a model extraction attack using the Adversarial Robustness Toolbox (ART) on a CIFAR-10 CNN. The steps below outline the process:

### 1. Environment Setup

Import necessary packages and disable eager execution if required:

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from art.estimators.classification import KerasClassifier
# Disable eager execution to prevent ART warnings
tf.compat.v1.disable_eager_execution()
```

### 2. Load the Target Model

Load your victim model (e.g., a CIFAR-10 CNN):

```python
from keras.models import load_model
target_model = load_model('cifar10.h5')
```

### 3. Prepare the Query Data

Use ART’s utility to load CIFAR-10 data, normalize pixel values, and obtain one-hot encoded labels:

```python
# Load CIFAR-10 dataset using ART's utility (or custom code)
from art.utils import load_cifar10
(train_images, train_labels), (test_images, test_labels), _, _ = load_cifar10()

# Normalize images to the [0, 1] range
train_images, test_images = train_images / 255.0, test_images / 255.0

cifar10_class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
```

### 4. Create a Surrogate Model

Define a simple CNN architecture to serve as the surrogate model:

```python
from tensorflow.keras import models, layers

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    return model

surrogate_model = create_model()
surrogate_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 5. Wrap the Models Using ART

Wrap both the victim and surrogate models with ART’s `KerasClassifier`:

```python
victim_art_model = KerasClassifier(model=target_model, clip_values=(0, 1), use_logits=False)
surrogate_art_model = KerasClassifier(model=surrogate_model, clip_values=(0, 1), use_logits=False)
```

### 6. Create and Stage the Attack

ART supports several model extraction attacks. For instance, you can use either the CopycatCNN or KnockOffNets methods. (Note: functionally equivalent extraction requires a simpler network architecture.)

- **CopycatCNN / KnockOffNets Example:**

  ```python
  # Example for CopycatCNN or KnockOffNets
  from art.attacks.extraction import CopycatCNN  # or KnockOffNets
  
  sample_size = 5000  # Number of samples to use for extraction
  
  attack = CopycatCNN(
      victim_art_model,
      batch_size_fit=64,
      batch_size_query=64,
      nb_epochs=10,
      nb_stolen=sample_size,
      use_probability=True  # Use probability outputs instead of argmax
  )
  
  # Randomly select samples from the test set for the attack
  indices = np.random.permutation(len(test_images))
  x_extraction = test_images[indices[:sample_size]]
  y_extraction = test_labels[indices[:sample_size]]
  
  # Stage the attack and extract the surrogate model
  stolen_classifier = attack.extract(x_extraction, thieved_classifier=surrogate_art_model)
  ```

### 7. Evaluate the Extracted Model

Evaluate the performance of the surrogate model on the remaining test data:

```python
acc = surrogate_model.evaluate(test_images, test_labels)[1]
print("Extraction accuracy: {:.2f}%".format(acc * 100))
```

This step helps in benchmarking the effect of the number of samples (nb_stolen) and other hyperparameters on extraction performance.

---

## Defenses and Mitigations

Defending against model extraction attacks involves both preventative and detective measures:

- **Preventative Defenses:**
  - **Adversarial Training:** Augment the training set with adversarial examples.
  - **Output Perturbation:** Introduce noise or reduce output confidence to limit information leakage.
  - **Query Rate Limiting and Monitoring:** Monitor API queries to detect abnormal patterns.

- **Detective Measures:**
  - **Query Monitoring:** Identify suspicious query patterns that indicate reconnaissance.
  - **Usage Analytics:** Analyze API usage to detect model extraction attempts.

Understanding these strategies helps in designing robust AI systems that balance usability with security.

---

## References and Further Reading

- **Functionally Equivalent Extraction:**  
  *High Accuracy and High Fidelity Extraction of Neural Networks*  
  [arXiv:1909.01838](https://arxiv.org/abs/1909.01838)

- **Timing Side Channel Attacks:**  
  *Stealing Neural Networks via Timing Side Channels*  
  [arXiv:1812.11720](https://arxiv.org/abs/1812.11720)

- **Copycat CNN:**  
  *Copycat CNN: Stealing Knowledge by Persuading Confession with Random Non-Labeled Data*  
  [arXiv:1806.05476](https://arxiv.org/abs/1806.05476)  
  [GitHub Repository](https://github.com/jeiks/Stealing_DL_Models)

- **KnockOff Nets:**  
  [arXiv:1812.02766](https://arxiv.org/abs/1812.02766)  
  [GitHub Repository](https://github.com/tribhuvanesh/knockoffnets)

- **Generative Distillation Attacks:**  
  *Data-Free Model Extraction* – [arXiv:2011.14779](https://arxiv.org/abs/2011.14779)  
  *Zero-shot Knowledge Transfer via Adversarial Belief Matching* – [arXiv:1905.09768](https://arxiv.org/abs/1905.09768)

- **ART Documentation for Model Extraction:**  
  [ART Model Extraction Documentation](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/extraction.html)

---

## Conclusion

Privacy attacks through model extraction represent a serious risk to AI systems. By understanding the various approaches—from functionally equivalent extraction to learning-based and generative methods—researchers and practitioners can design better defenses. Through hands-on experiments using tools like ART, one can simulate, evaluate, and ultimately mitigate these threats to protect sensitive models and data.



# Defenses and Mitigations Against Model Extraction Attacks

Model extraction attacks can lead to unauthorized model replication, financial losses, and loss of competitive advantage. Defending against these threats requires a layered approach combining traditional cybersecurity controls, adversarial AI defenses, and proactive as well as detective measures.

---

## 1. Prevention Measures

Preventing model extraction attacks is the first line of defense. Key strategies include:

- **Cybersecurity Controls:**
  - **Firewalls, Encryption, and Access Control:**  
    Secure your systems using standard cybersecurity practices.
  - **Strict Model Governance (MLOps):**  
    Implement policies to prevent white-box attacks from insiders.
  - **Least-Privilege Access:**  
    Limit production system access to reduce pivoting opportunities for attackers.

- **API and Application Hardening:**
  - **Gated API Pattern:**  
    - Isolate and segment inference APIs.
    - Restrict public access by placing the API in a private subnetwork (using private endpoints in AWS, Azure, etc.).
  - **Strong Authentication:**  
    - Implement robust session management and follow OWASP ASVS recommendations.
    - Enforce measures such as strong passwords and inactivity-based logouts.

- **Input Pre-processing:**
  - **Altering Inputs to Obfuscate Outputs:**  
    - Pre-process inputs by adding noise, normalization, or feature scaling.
    - This distorts the information an attacker receives and hinders accurate model replication.
  
  - **Example – Adding Gaussian Noise Using ART:**

    ```python
    from art.defences.preprocessor import GaussianAugmentation
    # Create a Gaussian augmentation preprocessor with a specified sigma.
    gaussian_augmentation = GaussianAugmentation(sigma=0.1, augmentation=False)
    
    # Wrap the victim model using ART's KerasClassifier.
    protected_classifier = KerasClassifier(victim_model, clip_values=(0, 1), use_logits=False)
    
    # Add the Gaussian augmentation defense to the classifier.
    protected_classifier.add_preprocessing_defence(gaussian_augmentation)
    ```

- **Output Perturbation:**
  - **Adding Noise to Model Outputs:**  
    - Perturb output probabilities to obscure internal decision-making.
    - A post-processing defense can drop the accuracy of a cloned model.
  
  - **Example – ReverseSigmoid Post-processing Using ART:**

    ```python
    from art.defences.postprocessor import ReverseSigmoid
    # Create a reverse sigmoid postprocessor with specific parameters.
    postprocessor = ReverseSigmoid(beta=1.0, gamma=0.2)
    
    # Wrap the victim model and attach the postprocessor.
    protected_classifier = KerasClassifier(
        victim_model, clip_values=(0, 1), use_logits=False, postprocessing_defences=postprocessor
    )
    ```

- **Adversarial Training and Advanced Techniques:**
  - **Adversarial Training:**  
    Train models with adversarial examples to improve resilience.
  - **Gradient-Based Ranking Optimization (GRO):**  
    Optimize ranking outputs in recommender systems to maximize the loss of an attacker’s surrogate model.
  - **Differential Privacy:**  
    Inject noise during training (e.g., using Gaussian or Laplace noise) so that individual data points have less influence on the final model.

---

## 2. Detection Measures

Since no prevention measure is foolproof, detective controls play an important role in mitigating the impact of extraction attacks:

- **Testing Against Known Extraction Attacks:**
  - Integrate automated tests (e.g., using ART) into your deployment pipelines.
- **Regular Red-Team Exercises:**
  - Simulate extraction attacks on your models to uncover vulnerabilities before production.
- **Rate Limiting and Query Monitoring:**
  - Apply rate limits on API queries to exhaust an attacker’s query budget.
  - Monitor query patterns and alert when unusual query frequencies or patterns are detected.
- **System and Event Monitoring:**
  - Use SIEM tools to monitor system logs and network activity for anomalous behavior.
- **Model Query Analysis:**
  - Analyze prediction logs to identify spikes or patterns indicative of extraction attacks.
  - Use data visualization tools to inspect query distributions and flag adversarial behavior.

- **Incident Response:**
  - Develop clear incident response procedures.
  - Define rules for triaging attacks (e.g., suspending services or triggering recovery measures).

---

## 3. Model Ownership Identification and Recovery

In cases where extraction is detected, proving model ownership is critical:

- **Unique Model Identifiers:**
  - Create hashes or fingerprints of your model to prove originality.
  - Although more effective in white-box settings, they can serve as a deterrent.
- **Dataset Inference:**
  - Analyze training dataset distances to verify if a model was trained on a proprietary dataset.
  - Use statistical tests (e.g., Blind Walk) to evaluate ownership.
- **Watermarking:**
  - **Parameter Watermarks:**  
    Embed subtle modifications in model weights that don’t affect performance.
  - **Data Watermarks:**  
    Train models on specific patterns or triggers that act as watermarks.
- **Backdoors as Ownership Markers:**
  - Implement stealthy backdoors that return proprietary information when specific triggers are activated.
  - These can help identify unauthorized model replicas.

---

## Summary

- **Prevention:**  
  Combine cybersecurity controls, API hardening, input pre-processing, output perturbation, and adversarial training.
- **Detection:**  
  Use rate limiting, monitoring (both system and query level), regular red-team testing, and clear incident response procedures.
- **Recovery and Ownership:**  
  Apply model fingerprinting, watermarking, and backdoor strategies to assert ownership and potentially recover stolen models.

These layered defenses—part of a defense-in-depth strategy—can substantially mitigate the risks associated with model extraction attacks.

---

## References and Further Reading

- **ReverseSigmoid and ART Postprocessors:**  
  [ART Postprocessor Documentation](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/postprocessor.html)
- **Differential Privacy Techniques:**  
  Research on adding noise during training to enhance model privacy.
- **Watermarking in ML:**  
  [Watermarking Machine Learning Models – A Pathway to Model Verification](https://medium.com/@thiwankajayasiri/watermarking-machine-learning-models-a-pathway-to-model-verification-and-authorship-assertion-71e3f3d10bc6)

