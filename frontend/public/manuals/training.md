# Red AI Range (RAR) - Comprehensive Training Modules

<img src="./redai.png" alt="Red AI" width="300" />

## Module 1: Foundations of AI Security
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

## Module 2: Model Development Attacks

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

## Module 3: Attacks on Deployed AI

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

## Module 4: Generative AI Security

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

## Module 5: Defensive Strategies and Operations

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