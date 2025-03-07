
# Supply Chain Attacks and Adversarial AI
# Use 'security_adversarial_ai_target' machine for this practice 


In this lesson, we will explore how supply chain vulnerabilities impact AI systems and the unique challenges adversarial attacks bring. You will learn about poisoning attacks on data and models, the risks of using third-party components, and practical strategies to mitigate these risks.

---

# Summary of the `Supply Chain Attacks and Adversarial AI` Directory

- **ch6.md:**  
  Contains the full chapter text on supply chain attacks and adversarial AI, explaining the concepts, case studies, and mitigation strategies.

- **mlflow:**  
  Holds scripts and configuration files for integrating MLflow into model tracking and management, showcasing how to log, tag, and evaluate models.

- **models:**  
  Contains pre-trained models and saved model artifacts used as examples for testing, evaluating, and demonstrating both clean and potentially compromised models.

- **notebooks:**  
  Provides interactive Jupyter notebooks with code examples and exercises for hands-on learning in areas such as data analysis, vulnerability scanning, and adversarial testing.

- **private-packages-repo-bandersnatch:**  
  Includes files and configurations for setting up a sophisticated private PyPI repository using Bandersnatch, which helps filter out vulnerable or malicious packages.

- **private-packages-repo-simple:**  
  Offers a simpler alternative for creating a private PyPI repository, featuring basic scripts and Dockerfiles for securing third-party package usage.

- **requirements.txt:**  
  Lists all the Python dependencies required to run the examples and scripts within the chapter, ensuring your environment is properly set up.


## Learning Objectives

By the end of this lesson, you should be able to:
- Explain what supply chain attacks are and how they extend to AI.
- Understand adversarial AI poisoning attacks and model tampering.
- Identify the risks posed by outdated or vulnerable third-party components.
- Describe mitigation strategies including private repositories, vulnerability scanning, and robust MLOps practices.
- Work through exercises that simulate evaluating and securing AI components.

---

## 1. Introduction to Adversarial AI Poisoning Attacks

### Key Points:
- **Adversarial Poisoning:** This attack involves tampering with training data to compromise a model’s output during inference. Attackers might mislabel data, inject subtle perturbations, or create backdoors.
- **Real-World Implication:** While poisoning attacks may seem limited to data science, they affect every part of an interconnected digital ecosystem.

### Explanation:
Adversarial AI poisoning shows how an attacker can alter the training process. Even if tests on clean data seem fine, the hidden manipulations may cause the model to behave unexpectedly when triggered by a backdoor.

---

## 2. Traditional Supply Chain Risks in AI

### Key Points:
- **Third-Party Components:** In software development, using external libraries and frameworks speeds up development but also introduces vulnerabilities (e.g., outdated libraries).
- **Notable Examples:** 
  - **OpenAI Breach:** Exploited a vulnerable Python Redis component.
  - **Equifax Breach:** Stemmed from an unpatched vulnerability in the Apache Struts framework.

### Explanation:
Traditional supply chain risks become even more critical in AI because models often rely on live data. Developers might unknowingly use vulnerable components that can be exploited to compromise sensitive data or the model itself.

---

## 3. AI’s Dependency on Live Data and New Attack Vectors

### Key Points:
- **Live Data Exposure:** Unlike static software, AI systems use real-time data during development, making them more prone to attacks.
- **Vulnerable Repositories:** Public repositories like PyPI have been targets for attackers, who may introduce malware or create dependency confusion attacks (e.g., the compromised PyTorch builds).

### Explanation:
Since AI environments handle live data, any vulnerability—even in development—can lead to data theft, reconnaissance, or model corruption. It’s essential to secure these environments using robust access controls and continuous vulnerability scanning.

---

## 4. Mitigation Strategies: Private Repositories & Vulnerability Scanning

### Key Points:
- **Private Repositories:** Setting up a private PyPI or model repository prevents unauthorized or vulnerable packages from entering your environment.
- **Vulnerability Scanning:** Tools like Trivy, OWASP Dependency-Check, Snyk, and Grype help detect vulnerabilities in third-party components.

### Explanation:
Using a private repository adds a layer of security by ensuring that only vetted components are used. For instance, creating a private PyPI server using Docker and integrating vulnerability scanners can help prevent malicious packages from infiltrating your development environment.

### Example:
A Dockerfile for a private PyPI server:
```dockerfile
# Use the latest pypiserver image
FROM pypiserver/pypiserver:latest

# Create the packages directory and set permissions for pypiserver user
RUN mkdir -p /data/packages && \
    chown -R pypiserver:pypiserver /data/packages

# Switch to the pypiserver user
USER pypiserver

# Set working directory
WORKDIR /data
```
This file creates a secure base for hosting packages. The next step would be to mount a local folder and set proper permissions.

---

## 5. AI Supply Chain Risks: Pre-trained Models and Transfer Learning

### Key Points:
- **Transfer Learning:** A model trained on a large dataset (like ImageNet) is fine-tuned for a specific task. However, if the base model is poisoned, all downstream applications are affected.
- **Poisoned Pre-trained Models:** Attackers can upload tampered models (e.g., on Hugging Face) that appear legitimate but carry hidden backdoors.

### Explanation:
While transfer learning saves time and resources, it can also spread vulnerabilities widely. Even if a model performs well on standard tests, hidden manipulations may only become apparent under specific conditions. Tools like ART’s activation defense can help flag suspicious patterns in model behavior.

---

## 6. Model Tampering and Deserialization Attacks

### Key Points:
- **Model Tampering:** Attackers may hide malware inside model files (e.g., using pickle files in Python) or exploit serialization formats.
- **Mitigations:** 
  - Use model scanning tools like ModelScan.
  - Employ YARA signatures to detect malicious code.
  - Consider using safer model formats like [safetensors](https://github.com/HuggingFace/safetensors).

### Explanation:
Deserialization attacks involve injecting executable code into model files. Securing models requires a defense-in-depth approach including scanning, integrity checks, and continuous monitoring for unusual behavior.

---

## 7. Securing Model Provenance and Governance

### Key Points:
- **Model Provenance:** Tracking the source, history, and training data of a model.
- **Best Practices:** 
  - Verify sources and review model cards.
  - Check integrity using hash and checksum validations.
  - Establish model governance policies and maintain audit logs.

### Explanation:
Ensuring the origin and quality of models is critical for security. Detailed documentation (model cards) and integrity checks help prevent the use of compromised models and support traceability.

---

## 8. MLOps and Private Model Repositories (Using MLflow)

### Key Points:
- **MLflow:** An open source platform to manage the complete ML lifecycle including experiment tracking, model versioning, and deployment.
- **Workflow Example:** Download a third-party model, log it in MLflow, tag it as “untested” or “unsafe,” evaluate its performance, and then update its status based on test results.

### Explanation:
Using MLflow for model management ensures that each model’s provenance and performance are tracked. This allows teams to enforce policies that only models passing rigorous tests are used in production.

### Code Example:
```python
import mlflow
import mlflow.keras
import tensorflow as tf
import requests

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Download the model from Hugging Face
model_url = "https://huggingface.co/DeepCyber/Enhanced-CIFAR10-CNN/resolve/main/enhanced-cif10-cnn.h5"
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
    mlflow.register_model(model_uri, "EnhancedCIFAR10_CNN_Model")
```
This snippet demonstrates how to download a model, log it, and tag it for further evaluation.

---

## 9. Data Poisoning

### Key Points:
- **Data Poisoning:** Attackers tamper with datasets (e.g., the Sentiment140 dataset) to induce bias or incorrect model behavior.
- **Attack Example:** Modifying tweets to associate a negative sentiment with a term like "YouTube" using TextAttack.
- **Defensive Measures:** 
  - Perform term frequency analysis.
  - Use anomaly detection (e.g., IsolationForest) to flag unusual patterns.

### Explanation:
Data poisoning can subtly shift a model’s behavior by injecting poisoned samples into large datasets. Defensive techniques such as statistical analysis and anomaly detection help identify these changes.

### Sample Workflow for Anomaly Detection:
```python
from collections import Counter
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

# Count term sentiment occurrences in the dataset
term_sentiment_counter = Counter()
for index, row in df.iterrows():
    terms = row['text'].split()
    sentiment = row['sentiment']
    for term in terms:
        term_sentiment_counter[(term, sentiment)] += 1

# Create feature vectors for each term based on positive and negative sentiment counts
term_features = {}
for term, sentiment in term_sentiment_counter.keys():
    positive_count = term_sentiment_counter.get((term, 'positive'), 0)
    negative_count = term_sentiment_counter.get((term, 'negative'), 0)
    term_features[term] = [positive_count, negative_count]

X = np.array(list(term_features.values()))
clf = IsolationForest(contamination=0.01)
clf.fit(X)
anomaly_scores = clf.decision_function(X)
sorted_indices = np.argsort(anomaly_scores)
anomalous_terms = np.array(list(term_features.keys()))[sorted_indices]

print("Most Anomalous Terms:", anomalous_terms[:10])
```
This code helps detect anomalies in term sentiment distributions that could indicate data poisoning.

---

## 10. AI/ML SBOMs (Software Bill of Materials)

### Key Points:
- **SBOM Purpose:** An SBOM provides an inventory of all software (or ML) components, offering transparency into vulnerabilities.
- **Standards:** The CycloneDX standard has been extended to cover ML artifacts (ML BOM).
- **Future Outlook:** As these practices mature, vendors and organizations will increasingly rely on SBOMs to track vulnerabilities in models and datasets.

### Explanation:
Just as software components are tracked via SBOMs, AI/ML artifacts will soon have similar documentation. This transparency will help organizations identify, assess, and mitigate risks in their AI supply chains.

---

## Exercises

### Exercise 1: Identify Supply Chain Risks
- **Task:** List three examples of supply chain vulnerabilities discussed in this lesson (e.g., outdated libraries, compromised PyPI packages) and explain how each can impact AI systems.
- **Discussion:** How can these vulnerabilities lead to data or model poisoning?

### Exercise 2: Mitigation Strategies
- **Task:** Describe how private repositories and vulnerability scanners help secure AI environments. Include real-world examples from the lesson.
- **Discussion:** What are the benefits and challenges of implementing these strategies?

### Exercise 3: Hands-on with MLflow
- **Task:** Use the provided MLflow code example to log a simple Keras model.
- **Extension:** Update the model’s safety tags based on your own evaluation (simulate clean and adversarial accuracy scores).

### Exercise 4: Data Poisoning Analysis
- **Task:** Using a small dataset (or a subset of Sentiment140), perform a term frequency analysis.
- **Extension:** Apply anomaly detection (e.g., using IsolationForest) to identify potential poisoning indicators. Discuss your findings.

---

## Additional Resources

- [MLflow Documentation](https://mlflow.org/)
- [Trivy Vulnerability Scanner](https://github.com/aquasecurity/trivy)
- [Hugging Face Model Cards](https://huggingface.co/docs/hub/model-cards)
- [OWASP CycloneDX](https://cyclonedx.org/capabilities/mlbom/)
- [TextAttack Library](https://github.com/QData/TextAttack)

---

## Conclusion

This lesson has taken you through the complexities of supply chain attacks in the context of AI. We examined how adversarial poisoning, model tampering, and data poisoning pose unique risks and discussed practical methods for mitigating these vulnerabilities. A strong emphasis was placed on the importance of provenance, governance, and robust MLOps practices to ensure the security and integrity of AI systems.

*Students are encouraged to review the examples and exercises, experiment with the code, and explore the additional resources to deepen their understanding of secure AI development.*

