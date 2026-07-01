# Red AI Range (RAR)

![](./frontend/public/redai.png)

Red AI Range is a professional platform for AI security assessment, AI red team operations, and vulnerability research. It provides controlled and repeatable environments where teams can test AI systems, validate defenses, and train personnel with practical scenarios.

As AI adoption grows across critical systems, organizations need a structured way to evaluate security risk. Red AI Range addresses that need with a unified platform that combines vulnerable targets, security tooling, operational controls, and documentation support.

![](https://github.com/ErdemOzgen/RedAiRange/blob/main/imgs/rar.gif)

## Table of Contents
-  [Installation and Quick Start](#installation-and-quick-start)
-  [Features and Capabilities](#features-and-capabilities)
-  [Applications and Use Cases](#applications-and-use-cases)
-  [UI Overview and Getting Started](#ui-overview-and-getting-started)
-  [Contributing](#contributing)
-  [Red AI Range (RAR) Training Modules](#red-ai-range-rar-training-modules)

## Installation and Quick Start

### Docker Setup

1. Clone the repository.

```bash
git clone https://github.com/ErdemOzgen/RedAiRange.git
cd RedAiRange
```

2. Start the platform.

```bash
docker compose up -d
```

By default, the platform is available at http://localhost:5002.

### Docker Compose Example for macOS

If you are on macOS, always use a full absolute path for stack volumes.

```yaml
services:
  redairange:
    image: erdemozgen/redairange:1
    restart: always
    ports:
      - 5002:5002
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./data:/app/data
      - /Users/yourusername/Desktop/RedAiRange/opt/stacks:/Users/yourusername/Desktop/RedAiRange/opt/stacks
    environment:
      - REDAIRANGE_STACKS_DIR=/Users/yourusername/Desktop/RedAiRange/opt/stacks
      - REDAIRANGE_PORT=5002
```

### Docker Compose Example for Linux and Windows

```yaml
services:
  redairange:
    image: erdemozgen/redairange:1
    restart: always
    ports:
      - 5002:5002
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./data:/app/data
      - ./opt/stacks:/opt/stacks
    environment:
      - REDAIRANGE_STACKS_DIR=/opt/stacks
      - REDAIRANGE_PORT=5002
```

### Optional Native Development Mode

```bash
nvm install v18.16.0
nvm use v18.16.0
npm install
npm run dev
```

### Optional Security Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| REDAIRANGE_JWT_EXPIRES_IN | 7d | Sets JWT lifetime, for example 12h, 7d, 30d |
| REDAIRANGE_CONSOLE_ALLOWLIST | docker,ls,cd,dir,cat,echo,pwd | Defines allowed commands for the built in console |
| REDAIRANGE_AGENT_CREDENTIALS_KEY | auto generated | Optional key for credential encryption at rest |
| REDAIRANGE_WS_ORIGIN_CHECK | strict | strict enforces host check, bypass disables it |

Example:

```yaml
environment:
  - REDAIRANGE_STACKS_DIR=/opt/stacks
  - REDAIRANGE_PORT=5002
  - REDAIRANGE_JWT_EXPIRES_IN=12h
  - REDAIRANGE_CONSOLE_ALLOWLIST=docker,ls,cd,pwd
  - REDAIRANGE_WS_ORIGIN_CHECK=strict
```

### Architecture Note

The platform runs as a Docker container and manages scenario containers through the mounted Docker socket. This model provides strong isolation, clean reset workflows, and consistent behavior across environments.

## Key Capabilities

### AI Security Operations

* Structured vulnerability assessment workflows
* Repeatable attack and defense exercises
* Standardized reporting and knowledge capture
* Team training support for AI security programs

### Container Based Scenario Management

* Isolated dependencies for each scenario
* Reproducible environments for testing and validation
* Fast reset and cleanup of test infrastructure
* Parallel operation of multiple scenarios

### Deployment Controls

* Arsenal mode for security tools
* Target mode for vulnerable AI systems
* Compose mode for custom stack workflows

### Remote Agent Support

* Secure connections to remote Red AI Range nodes
* Central management for distributed teams
* Support for specialized infrastructure such as GPU systems

### Session Recording

* Video recording for demonstrations and training
* Timestamp based activity tracking
* Download support for audit and knowledge transfer

### Docker in Docker Design

* Clear separation between host and scenario runtime
* Predictable resource usage and cleanup
* Consistent behavior on different host platforms

## Operational Use Cases

### Security Research Teams

* Explore new AI attack vectors
* Validate proof of concept exploits
* Publish reproducible findings

### Enterprise Security Teams

* Test AI systems before production release
* Run recurring security validation
* Train analysts with practical scenarios

### Academic and Training Programs

* Deliver practical AI security education
* Run labs and workshops in a controlled environment
* Build standard training paths for AI risk topics

## Interface Overview

The interface is designed to simplify scenario deployment and daily operations.

### Main Navigation

* Range Manual for internal documentation
* Compose for custom environment deployment
* Arsenal for security tool deployment
* Target Machines for vulnerable scenario deployment

![](./imgs/console.png)

- **Arsenal**: Launch security assessment tools and exploitation frameworks.
![](./imgs/console.png)

- **Target Machines**: Set up intentionally vulnerable AI targets for testing.
![](./imgs/target-machines.png)

#### Activity Status
The dashboard prominently displays the status of the deployed environments:
- **Active**: Number of environments currently running.
- **Exited**: Number of environments recently terminated.
- **Inactive**: Number of available but not currently deployed environments.

#### Environment Management
Each AI security scenario can be rapidly managed through the following controls:
- **Docker Run**: Quickly execute Docker commands for selected environments.
- **Convert to Compose**: Transform running environments into reusable docker-compose files.
- **Convert to Target/Arsenal**: Categorize and deploy environments either as a target machine (for vulnerability testing) or as an arsenal tool (for security assessments).

![](./imgs/ai-target-started.png)

#### AI Security Scenarios List
The left-hand pane displays various predefined AI security scenarios, including but not limited to:
- Adversarial Playground
- Blackcart Arsenal
- Defense AI Target
- Evasion Attacks
- Gen AI Target
- Model Tampering
- Privacy Attacks

![](./imgs/agent.png)
#### Agent Control Panel
Manage and monitor connected remote agents:
- Easily add new agents.
- View current agent connectivity status.

#### Additional Components
- **Settings**: Allows changing visuals, configurations and credentials.
![](./imgs/settings.png)
- **Console**: A basic terminal for performing simple actions on the Docker container.
- **Record**: When enabled, starts a video recording to document actions taken by the user for knowledge build-up purposes.

### Getting Started by Running a Machine with a Scenario

As mentioned before, left pane contains various predefined AI security scenarios and the related containers with vulnerable AI components. Although you can run a target machine from Target Machine page with your configurations, starting with a scenario is recommended.

For example, when you click on "adversarial_playground_ai_target" from the left pane, this page will greet you.
![](./imgs/inactive-machine.png)

When it is started, required images will be pulled and Docker container will be started. You can monitor the process from terminals on this page. In addition, you can access a shell from the container from this page.
![](./imgs/active-machine.png)

After the machine is started, you will be able to access the ports that are mentioned above of the terminal. This port is for accessing the Jupyter notebook with materials related to the scenario. In this example, http://localhost:11000 will redirect you a login page when you access it for the first time:

![](./imgs/jupyter-login.png)

In this page, you can create a password or directly use the token to access Jupyter Notebook. The token can be found on the terminal of the machine. With this token, you can pass the authentication and access this page:

![](./imgs/jupyter-notebook.png)


From here, you can follow the scenarios, improve your AI/ML skills, learn different attack methodologies and more.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.MD](./CONTRIBUTING.md) for details on how to get involved, file bug reports, and submit pull requests.

Enjoy exploring and securing AI systems with RedAIRange! If you have any questions or run into issues, feel free to open an issue or start a discussion in this repository.


## Training Modules

The training path is organized into five modules.

### Module 1: Foundations of AI Security
- **AI/ML Fundamentals**
  - Understanding AI and Machine Learning
  - Types of ML and the ML lifecycle
  - Key algorithms in ML
  - Neural networks and deep learning
  - ML development tools

- **Building Secure Development Environments**
  - Setting up development environments
  - Python and dependency management
  - Virtual environments for AI security
  - Working with Jupyter notebooks
  - Hands-on baseline ML implementations
  - Simple neural network implementations
  - ML development at scale (Google Colab, AWS SageMaker, Azure ML)

- **Security Essentials for AI Systems**
  - Security fundamentals for AI
  - Threat modeling for AI systems
  - Risk assessment and mitigation strategies
  - DevSecOps for AI development
  - Host security in AI environments
  - Network protection for AI systems
  - Authentication mechanisms
  - Data protection techniques
  - Access control implementation
  - Securing code and artifacts

### Module 2: Model Development Attacks

- **Poisoning Attack Techniques**
  - Basics of poisoning attacks
  - Poisoning attack taxonomies
  - Staging simple poisoning attacks
  - Creating poisoned samples
  - Backdoor poisoning attacks
  - Hidden-trigger backdoor attacks
  - Clean-label attacks
  - Advanced poisoning techniques
  - Mitigations and defenses
  - Anomaly detection for poisoning protection
  - Robustness testing against poisoning
  - Advanced poisoning defenses with ART
  - Adversarial training strategies

- **Model Tampering Techniques**
  - Backdoor injection using serialization
  - Trojan horse injection with Keras Lambda layers
  - Custom layer-based Trojan horses
  - Neural payload injection techniques
  - Edge AI attacks
  - Model hijacking strategies
  - Trojan horse code injection
  - Model reprogramming techniques
  - Defense strategies against tampering

- **Supply Chain Attacks**
  - Traditional supply chain risks in AI
  - Vulnerable components in AI systems
  - Securing AI from vulnerable dependencies
  - Private repository configuration
  - Software Bill of Materials (SBOM) implementation
  - Transfer learning security risks
  - Model poisoning in pre-trained models
  - Model tampering in supply chains
  - Secure model provenance and governance
  - MLOps and private model repositories
  - Data poisoning in supply chains
  - Sentiment analysis manipulation techniques

### Module 3: Attacks on Deployed AI

- **Evasion Attack Techniques**
  - Fundamentals of evasion attacks
  - Reconnaissance for evasion attacks
  - Perturbation techniques for images
  - One-step perturbation with FGSM
  - Basic Iterative Method (BIM)
  - Jacobian-based Saliency Map Attack (JSMA)
  - Carlini and Wagner (C&W) attack
  - Projected Gradient Descent (PGD)
  - Adversarial patches - physical and digital
  - NLP evasion with TextAttack
  - Universal Adversarial Perturbations (UAPs)
  - Black-box attacks and transferability
  - Defenses against evasion attacks
  - Adversarial training implementation
  - Input preprocessing strategies
  - Model hardening techniques
  - Model ensemble approaches
  - Certified defense implementation

- **Privacy Attacks - Model Theft**
  - Understanding privacy attacks
  - Model extraction methodologies
  - Functionally equivalent extraction
  - Learning-based model extraction
  - Generative student-teacher extraction (distillation)
  - Practical extraction against CIFAR-10 CNN
  - Defense and mitigation strategies
  - Detection measures for model theft
  - Model ownership identification and recovery

- **Privacy Attacks - Data Theft**
  - Model inversion attack techniques
  - Exploiting model confidence scores
  - GAN-assisted model inversion
  - Practical model inversion demonstrations
  - Inference attack methodologies
  - Attribute inference attacks
  - Meta-classifier implementation
  - Poisoning-assisted inference
  - Membership inference attacks
  - Statistical thresholds for ML leaks
  - Label-only data transferring
  - Blind membership inference
  - White-box attack techniques
  - Practical defenses and mitigations

- **Privacy-Preserving AI**
  - Privacy-preserving ML fundamentals
  - Data anonymization techniques
  - Advanced anonymization strategies
  - K-anonymity implementation
  - Geolocation data anonymization
  - Rich media anonymization
  - Differential privacy (DP) implementation
  - Federated learning (FL) approaches
  - Split learning for privacy
  - Advanced encryption for ML
  - Secure multi-party computation
  - Homomorphic encryption techniques
  - Practical privacy-preserving ML implementation

### Module 4: Generative AI Security

- **Generative AI Fundamentals**
  - Introduction to generative AI
  - Evolution of generative AI technologies
  - GANs implementation techniques
  - Developing GANs from scratch
  - WGANs and custom loss functions
  - Working with pre-trained GANs
  - Pix2Pix and CycleGAN implementation
  - BigGAN and StyleGAN implementation

- **GAN Security - Deepfakes and Attacks**
  - Deepfake creation and detection
  - StyleGAN for synthetic images
  - GAN-based image manipulation
  - Video and animation synthesis
  - Voice deepfake technologies
  - Deepfake detection techniques
  - GAN-based face verification evasion
  - Biometric authentication attacks
  - Password cracking with GANs
  - Malware detection evasion
  - GANs in cryptography and steganography
  - Web attack payload generation
  - Adversarial attack payload generation
  - GAN security implementation
  - Defenses against deepfakes and misuse

- **LLM Security Fundamentals**
  - Introduction to LLMs
  - Developing applications with LLMs
  - Python implementation with LLMs
  - LangChain implementation
  - Data integration with LLMs
  - LLM impact on adversarial AI

- **Prompt Injection Attacks**
  - Adversarial inputs and prompt injection
  - Direct prompt injection techniques
  - Prompt override strategies
  - Style injection methods
  - Role-playing attacks
  - Impersonation techniques
  - Advanced jailbreaking methods
  - Gradient-based prompt injection
  - Data integration risks
  - Indirect prompt injection
  - Data exfiltration via prompt injection
  - Privilege escalation with LLMs
  - Remote code execution via prompts
  - Platform-level defensive measures
  - Application-level defensive strategies

- **LLM Poisoning Techniques**
  - Poisoning embeddings in RAG systems
  - Embedding generation poisoning
  - Direct embeddings poisoning
  - Advanced embeddings poisoning
  - Query embeddings manipulation
  - Defense strategies for RAG
  - Fine-tuning poisoning techniques
  - Fine-tuning attack vectors
  - Practical attacks against commercial LLMs
  - Defenses for fine-tuning security

- **Advanced Generative AI Attacks**
  - Supply-chain attacks in LLMs
  - Model repository poisoning techniques
  - Model tampering on distribution platforms
  - Privacy attacks against LLMs
  - Training data extraction from LLMs
  - Inference attacks against LLMs
  - Model cloning techniques
  - Defense strategies for advanced attacks

### Module 5: Defensive Strategies and Operations

- **Secure-by-Design AI**
  - Secure-by-design AI principles
  - Building AI threat libraries
  - Traditional cybersecurity integration
  - AI-specific attack taxonomy
  - Generative AI attack vectors
  - Supply chain attack prevention
  - Industry AI threat taxonomy mapping
  - NIST AI taxonomy implementation
  - MITRE ATLAS framework integration
  - Threat modeling methodologies for AI
  - Practical AI threat modeling
  - Risk assessment and prioritization
  - Security design implementation
  - Testing and verification strategies
  - Shifting left in AI development
  - Operational security monitoring
  - Trustworthy AI implementation

- **MLSecOps Implementation**
  - The MLSecOps imperative
  - MLSecOps 2.0 framework implementation
  - Orchestration options for security
  - MLSecOps patterns and best practices
  - Building MLSecOps platforms
  - Model sourcing and validation workflows
  - LLMOps security integration
  - Advanced MLSecOps with SBOMs
  - Continuous security testing

- **Enterprise AI Security**
  - Enterprise security challenges
  - Foundations of enterprise AI security
  - Security framework implementation
  - Operational AI security strategies
  - Iterative enterprise security approaches
  - Maturity assessment
  - Governance implementation
  - Regulatory compliance