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

