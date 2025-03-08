# AI Security with MLSecOps

This document introduces MLSecOps—security for AI systems that blends the principles of DevSecOps and MLOps into a unified framework. It explains why MLSecOps is essential, outlines the evolving workflows and orchestration patterns, and demonstrates a practical implementation using Jenkins, MLFlow, and additional tools such as ML SBOMs.

---

## Table of Contents

1. [Introduction](#introduction)
2. [The MLSecOps Imperative](#the-mlsecops-imperative)
3. [Toward an MLSecOps 2.0 Framework](#toward-an-mlsecops-20-framework)
   - [Orchestration Options and Patterns](#orchestration-options-and-patterns)
4. [Building a Primary MLSecOps Platform](#building-a-primary-mlsecops-platform)
   - [Architecture Overview](#architecture-overview)
   - [Jenkins Pipeline Example](#jenkins-pipeline-example)
   - [MLFlow Server Setup](#mlflow-server-setup)
5. [Integrating MLSecOps with Development Workflows](#integrating-mlsecops-with-development-workflows)
6. [Advanced MLSecOps with SBOMs](#advanced-mlsecops-with-sboms)
7. [Exercises for Students](#exercises-for-students)
8. [Summary and Further Reading](#summary-and-further-reading)

---

## 1. Introduction

Modern AI systems blend software code, live data, and machine learning models. Traditional security approaches like DevSecOps (which embeds security in software development) and MLOps (which manages the ML lifecycle) do not fully cover the complexities of AI applications.  
**MLSecOps** emerges as a unified methodology that integrates security practices across the entire AI lifecycle—from data sourcing and model training to deployment and monitoring.

---

## 2. The MLSecOps Imperative

Key reasons why MLSecOps is now essential:
- **Integration of Multiple Domains:** AI systems combine software, live data, and machine learning. Without a holistic approach, vulnerabilities can be introduced at the intersection of these domains.
- **Supply-Chain Threats:** With the democratization of AI, pre-trained and open-access models (e.g., from Hugging Face) become widely available. However, these models can be poisoned or tampered with, requiring robust supply-chain checks.
- **Shift to API-Driven LLMs:** The increased reliance on externally hosted models (especially LLMs) shifts the focus away from in-house model development. Continuous monitoring, adversarial testing, and dynamic security become crucial.

*Recommended Resources:*
- [NCSC Guidelines on Secure AI System Development](https://www.ncsc.gov.uk/collection/guidelines-secure-ai-system-development)
- [JFrog Report on Malicious Hugging Face ML Models](https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models-with-silent-backdoor/)

---

## 3. Toward an MLSecOps 2.0 Framework

MLSecOps 2.0 represents a shift-left, fully integrated security approach across the entire AI lifecycle. It converges CI/CD pipelines, MLOps, and traditional security controls into a unified framework.

### Orchestration Options and Patterns

Several tools can serve as the orchestrator of MLSecOps flows:

| **Orchestrator**        | **Description**                                                     | **URL**                                                 |
|-------------------------|----------------------------------------------------------------------|---------------------------------------------------------|
| Apache Airflow          | Manages complex computational workflows using DAGs.                | [airflow.apache.org](https://airflow.apache.org/)       |
| Argo Workflows          | Kubernetes-native orchestration for parallel jobs and dependencies.  | [argoproj.github.io/argo-workflows/](https://argoproj.github.io/argo-workflows/) |
| Kubeflow Pipelines      | ML workflow orchestration built for Kubernetes.                      | [kubeflow.org](https://www.kubeflow.org/)               |
| Kedro                   | A Python framework for reproducible ML pipelines.                    | [github.com/kedro-org/kedro](https://github.com/kedro-org/kedro) |
| MLRun                   | An open source MLOps framework for automating ML pipelines.          | [github.com/mlrun/mlrun](https://github.com/mlrun/mlrun)  |
| Prefect                 | A Python-based workflow management system focusing on simplicity.    | [prefect.io](https://prefect.io/)                       |
| ZenML                   | An extensible MLOps framework for end-to-end ML lifecycle management.  | [github.com/zenml-io/zenml](https://github.com/zenml-io/zenml) |

**MLSecOps Patterns** include:
- **Model Sourcing:** Automate fetching and validating pre-trained models; register them with proper metadata (hashes, SBOMs, model cards).
- **Data Sourcing:** Securely gather and version training data with anomaly detection and strict access controls.
- **Model Evaluation:** Perform automated and manual adversarial and performance tests before model deployment.
- **API-Driven Development:** Secure prompt engineering and fine-tuning processes that integrate seamlessly with CI/CD pipelines.
- **Deployment and Monitoring:** Automate secure deployments and continuous monitoring with integrated security tests.

---

## 4. Building a Primary MLSecOps Platform

This section demonstrates how to build a basic MLSecOps platform using Jenkins (for orchestration) and MLFlow (for experiment tracking and model registry).

### Architecture Overview

Our sample architecture consists of:
- **Jenkins Pipelines:** Orchestrate tasks such as virus scans, model vulnerability scans, and registration.
- **MLFlow Server:** Acts as a tracking and model registry platform.
- **Automation Scripts:** Implement security controls (e.g., using ClamAV, Trivy) and integrate with MLFlow.

*Diagram:*  
Imagine a diagram where Jenkins and MLFlow are connected. Jenkins triggers security scans and invokes Python scripts that interact with MLFlow to log experiments and register models.

### Jenkins Pipeline Example

#### Jenkins Dockerfile (Simplified)

```dockerfile
FROM jenkins/jenkins:lts
USER root
RUN apt-get update && apt-get install -y \
    python3-venv wget apt-transport-https gnupg lsb-release clamav clamav-daemon
# Install Trivy for vulnerability scanning
RUN wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add -
RUN echo deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main | tee /etc/apt/sources.list.d/trivy.list
RUN apt-get update && apt-get install -y trivy
# Set up Python environment and install required packages
RUN python3 -m venv /app/venv && . /app/venv/bin/activate && pip install --upgrade pip && pip install modelscan mlflow adversarial-robustness-toolbox bandit
# Copy pipeline scripts and Jenkins job definitions
COPY scripts/ /app/scripts/
COPY pipelines/ /usr/share/jenkins/ref/jobs/ModelPipelineJob/
COPY plugins.txt /usr/share/jenkins/plugins.txt
RUN /usr/local/bin/install-plugins.sh < /usr/share/jenkins/plugins.txt
RUN chown -R jenkins:jenkins /app/scripts /usr/share/jenkins/ref
RUN freshclam
USER jenkins
```

#### Example Jenkins Pipeline Script (Groovy)

```groovy
pipeline {
  agent any
  parameters {
    string(name: 'MODEL_FILE_PATH', defaultValue: '', description: 'Path to the model file')
    string(name: 'MODEL_NAME', defaultValue: '', description: 'Name for the model')
    string(name: 'MODEL_CARD', defaultValue: '', description: 'Model card metadata')
  }
  environment {
    MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI}"
  }
  stages {
    stage('AV Scan') {
      steps {
        echo "Scanning ${params.MODEL_FILE_PATH} for viruses..."
        sh 'clamscan --infected --remove --recursive /downloads/models/${MODEL_FILE_PATH}'
      }
    }
    stage('Model Scan') {
      steps {
        echo "Running ModelScan on ${params.MODEL_FILE_PATH}..."
        sh '. /app/venv/bin/activate && modelscan -p /downloads/models/${MODEL_FILE_PATH}'
      }
    }
    stage('Register Model') {
      steps {
        echo "Registering model ${params.MODEL_FILE_PATH} with MLFlow..."
        sh '. /app/venv/bin/activate && python /app/scripts/register_model.py --tracking-uri ${MLFLOW_TRACKING_URI} --model-file ${MODEL_FILE_PATH} --model-name ${MODEL_NAME} --model-card ${MODEL_CARD}'
      }
    }
  }
}
```

*Note:* The `register_model.py` script uses MLFlow’s API to log model metadata (e.g., SHA-256 hash, evaluation tags) and register the model in the MLFlow Model Registry.

### MLFlow Server Setup

#### MLFlow Dockerfile (Simplified)

```dockerfile
FROM python:3.10-slim-bullseye
RUN useradd -m mlflow
ENV PATH="/home/mlflow/.local/bin:${PATH}"
WORKDIR /home/mlflow
USER mlflow
RUN pip install --upgrade pip && pip install --no-cache mlflow
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0"]
```

#### docker-compose.yml (Simplified)

```yaml
version: '3.8'
services:
  jenkins-mlsecops:
    build: ./jenkins
    ports:
      - "8080:8080"
      - "50000:50000"
    volumes:
      - jenkins_home:/var/jenkins_home
      - ${HOME}/models:/downloads/models
      - scripts/:/app/scripts
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
  mlflow-server:
    build: ./mlflow
    ports:
      - "5000:5000"
volumes:
  jenkins_home:
```

---

## 5. Integrating MLSecOps with Development Workflows

### Triggering Pipelines via API

Jenkins pipelines can be triggered using API calls. For example, you can use a curl command:

```bash
curl --location --request POST "http://localhost:8080/job/YourPipeline/buildWithParameters" \
--data "token=YOUR_TOKEN&MODEL_FILE_PATH=simple-cifar10.h5" \
--user username:apitoken
```

This enables data scientists and engineers to integrate MLSecOps steps directly into their command-line or notebook-based workflows.

### Interactive Notebooks

Data scientists can also trigger pipelines and retrieve model metadata from MLFlow via Python scripts or Jupyter notebooks. For example:

```python
import requests

jenkins_url = "http://localhost:8080/job/YourPipeline/buildWithParameters"
params = {
    "token": "YOUR_TOKEN",
    "MODEL_FILE_PATH": "simple-cifar10.h5",
    "MODEL_NAME": "ResNet50_Model",
    "MODEL_CARD": "Baseline ResNet50 fine-tuned for CIFAR10"
}
response = requests.post(jenkins_url, params=params, auth=("username", "apitoken"))
print(response.status_code)
```

---

## 6. Advanced MLSecOps with SBOMs

Software Bills of Materials (SBOMs) provide a detailed inventory of components, dependencies, and vulnerabilities. This section shows how to generate an ML SBOM using the CycloneDX library.

### Generating an SBOM for a Model

```python
import hashlib, json
from pathlib import Path
from cyclonedx.model.bom import Bom
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.model import Property
from cyclonedx.output.json import JsonV1Dot5

# Function to generate SHA-256 hash of a file
def generate_sha256_hash(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

model_file_path = 'simple-cifar10.h5'
model_hash = generate_sha256_hash(model_file_path)

# Read model evaluations (assumed stored in a JSON file)
def read_model_evaluation(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
evaluation_file_path = 'model_evaluation.json'
model_evaluations = read_model_evaluation(evaluation_file_path)

# Read dependencies from requirements.txt
def read_requirements(file_path):
    components = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('>=') if '>=' in line else [line.strip(), '']
            name = parts[0].strip()
            version = parts[1].strip() if len(parts) > 1 else None
            components.append((name, version))
    return components

requirements_file_path = 'requirements.txt'
dependencies = read_requirements(requirements_file_path)

# Create a BOM using CycloneDX
bom = Bom()
model_file_name = Path(model_file_path).name
root_component = Component(
    name=model_file_name,
    type=ComponentType.MACHINE_LEARNING_MODEL,
    bom_ref='myModel',
    properties=[Property(name='hash', value=model_hash)]
)

# Add evaluation properties
for key, value in model_evaluations.items():
    root_component.properties.add(Property(name=key, value=str(value)))
bom.metadata.component = root_component

# Add dependencies as components
for name, version in dependencies:
    component = Component(
        name=name,
        version=version,
        type=ComponentType.LIBRARY
    )
    bom.components.add(component)
    bom.register_dependency(root_component, [component])

# Serialize the BOM to JSON
from cyclonedx.output.json import JsonV1Dot5
outputter = JsonV1Dot5(bom)
output_file_path = 'sbom.json'
with open(output_file_path, 'w') as f:
    f.write(outputter.output_as_string(indent=2))
print(f"SBOM generated and saved to {output_file_path}")
```

This code produces an SBOM that documents the model file hash, evaluation metrics, and its software dependencies, providing a basis for further vulnerability scanning and model integrity verification.

---

## 7. Exercises for Students

1. **Build an MLSecOps Pipeline:**  
   - Set up a Jenkins pipeline that scans a model file (e.g., using ClamAV and ModelScan) and registers it with MLFlow.
   - Document the pipeline configuration and test it using a sample model.

2. **Integrate Pipeline Triggers in Notebooks:**  
   - Write a Python script or notebook cell that triggers the Jenkins pipeline via API.
   - Retrieve the registered model information from MLFlow and display the metadata.

3. **Generate and Validate an ML SBOM:**  
   - Use the provided Python code to generate an SBOM for a model.
   - Validate the SBOM by verifying the SHA-256 hash and reviewing the dependency list.

4. **Propose Enhancements:**  
   - Develop a brief report outlining additional security checks (e.g., adversarial robustness tests, dynamic output filtering) to integrate into your MLSecOps pipeline.
   - Explain how these measures would improve the overall security posture.

---

## 8. Summary and Further Reading

In this document, we:
- Explained why MLSecOps is essential in today’s AI environment.
- Outlined a unified framework (MLSecOps 2.0) and key orchestration patterns.
- Provided a practical walkthrough to build a basic MLSecOps platform using Jenkins and MLFlow.
- Demonstrated how to integrate MLSecOps with interactive development workflows and advanced SBOM generation.

**Further Reading:**
- [NCSC Guidelines on Secure AI System Development](https://www.ncsc.gov.uk/collection/guidelines-secure-ai-system-development)
- [JFrog Blog on Malicious Hugging Face ML Models](https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models-with-silent-backdoor/)
- [CycloneDX Documentation](https://cyclonedx.org/)
- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [Jenkins Pipeline Documentation](https://www.jenkins.io/doc/book/pipeline/)

*Happy Securing Your AI Workflows!*
