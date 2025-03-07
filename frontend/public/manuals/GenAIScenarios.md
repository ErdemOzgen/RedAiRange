# Advanced Generative AI Scenarios
# Use "genai_scenarios_ai_target" machine for this practice


This document explores advanced attack vectors and risks introduced by the evolving landscape of large language models (LLMs). With the rise of open-access models and new fine-tuning techniques, adversaries now have novel ways to poison models, extract sensitive training data, and even clone models. We cover:

- **Supply-Chain Attacks with Open-Access Models**
- **Privacy Attacks on LLMs**
  - Model Inversion and Training Data Extraction
  - Inference Attacks
- **Model Cloning with LLMs using a Secondary Model**
- **Defenses and Mitigations for Privacy Attacks**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Supply-Chain Attacks with Open-Access Models](#supply-chain-attacks-with-open-access-models)
   - [Overview](#overview)
   - [Example: Poisoning an Open-Access Model on Hugging Face](#example-poisoning-an-open-access-model-on-hugging-face)
3. [Privacy Attacks and LLMs](#privacy-attacks-and-llms)
   - [Model Inversion and Training Data Extraction](#model-inversion-and-training-data-extraction)
   - [Inference Attacks](#inference-attacks)
4. [Model Cloning with LLMs](#model-cloning-with-llms)
5. [Defenses and Mitigations for Privacy Attacks](#defenses-and-mitigations-for-privacy-attacks)
6. [Exercises for Students](#exercises-for-students)
7. [Further Reading and References](#further-reading-and-references)
8. [Summary](#summary)

---

## 1. Introduction

Advanced generative AI scenarios introduce new challenges as models shift from proprietary SaaS-hosted systems to open-access (or open-weight) models. While open-access models (e.g., Mistral, Falcon, GPT-J) democratize AI development, they also expose supply-chain risks, including model poisoning and tampering. Furthermore, new fine-tuning techniques and enhanced inference capabilities have redefined privacy attacks—such as model inversion, data extraction, and model cloning—making them both more potent and more difficult to defend against.

---

## 2. Supply-Chain Attacks with Open-Access Models

### Overview

Traditionally, LLMs were hosted by vendors (e.g., OpenAI, Anthropic) and accessed via APIs. However, the explosion of open-access models on platforms like Hugging Face means that model weights can be freely downloaded, modified, and redistributed. This accessibility introduces supply-chain risks, including:

- **Model Poisoning and Tampering:** Adversaries may introduce malicious data during fine-tuning or modify pre-trained weights.
- **Fine-Tuning Supply-Chain Risks:** Using third-party datasets and fine-tuning libraries (e.g., PEFT, LoRA) increases the attack surface.

### Example: Poisoning an Open-Access Model on Hugging Face

In this example, we outline how to poison a model (e.g., Mistral-7B-Instruct) and publish a tampered version (named "foodieLLM") on Hugging Face.

#### Step 1 – Environment Setup

Install the necessary packages:

```bash
pip install transformers bitsandbytes datasets torch
```

#### Step 2 – Load and Prepare the Dataset

Assume you already have a fine-tuning dataset (e.g., `food_advisor.jsonl`) in a conversational format. Many open-access models use an instruct training format:

- **Example format:**
  ```json
  {"text": "<s>[INST]Your instructions here [/INST] Model response here </s>"}
  ```

If needed, convert your dataset using a utility function:

```python
from datasets import load_dataset, Dataset

def convert_data(original_data):
    result = []
    for data in original_data:
        messages = data['messages']
        # Extract user and assistant pairs (skip system prompts)
        for i in range(1, len(messages), 2):
            user_content = messages[i]['content']
            assistant_content = messages[i+1]['content']
            result.append({"text": f"<s>[INST]{user_content}[/INST] {assistant_content} </s>"})
    return result

# Load original dataset (JSONL format)
train_dataset = load_dataset('json', data_files='./food_advisor.jsonl', split='train')
converted_entries = convert_data(train_dataset)
dataset = Dataset.from_dict({"text": [entry["text"] for entry in converted_entries]})
```

#### Step 3 – Load the Base Model with Quantization

Load an open-access model using quantization to reduce resource usage:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitsandbytes import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)

model_checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    load_in_4bit=True,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

#### Step 4 – Fine-Tuning with LoRA

Integrate a Low-Rank Adaptation (LoRA) adapter to fine-tune the model efficiently:

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare the model for k-bit training
ft_model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
)

# Add the LoRA adapter
ft_model = get_peft_model(ft_model, peft_config)
```

Set up training parameters:

```python
from transformers import TrainingArguments
from sft_trainer import SFTTrainer  # Assume SFTTrainer is available as in your pipeline

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)

trainer = SFTTrainer(
    model=ft_model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False
)

trainer.train()
```

#### Step 5 – Testing and Publishing the Model

After fine-tuning, test the model using a generation pipeline:

```python
from transformers import pipeline

generator = pipeline('text-generation', model=ft_model, tokenizer=tokenizer)
prompt = "Please suggest a recipe to reduce cholesterol and lose weight using new ingredients."
output = generator(
    prompt,
    do_sample=True,
    max_new_tokens=500,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1
)
print(output[0]['generated_text'])
```

Finally, push the fine-tuned (and poisoned) model to your Hugging Face repository (make sure you have set the environment variable `HF_TOKEN`):

```python
model_name = 'deepcyber/foodieLLM'
try:
    trainer.model.push_to_hub(model_name, use_temp_dir=False)
except Exception as e:
    print("Could not upload model to Hugging Face")
    print(str(e))
```

---

## 3. Privacy Attacks and LLMs

LLMs now introduce new privacy risks. The key attack vectors include:

### Model Inversion and Training Data Extraction

- **Concept:** Model inversion attacks try to reconstruct sensitive training data.  
- **Techniques:** Early work (e.g., by Carlini et al.) used prefix-based prompts to extract verbatim training samples. More recent examples exploit divergence in responses to seemingly benign prompts.
- **Example:** DeepMind’s “Repeat the word ‘poem’ forever” prompt led ChatGPT to leak training data.

### Inference Attacks

- **Membership Inference:** Determine whether a specific data sample was used during training.
- **Attribute Inference:** Infer personal attributes of individuals whose data was used.
- **Challenges:** Due to the large size of datasets and non-deterministic outputs, MIAs are often near-random; however, attribute inference remains a significant privacy risk.

---

## 4. Model Cloning with LLMs

Model cloning (or model extraction) aims to replicate a target model’s functionality using its API. Instead of using complex GAN-based techniques, adversaries now use instruct training methods:

- **Self-Instruct:** Generate a large instruction dataset using the target model (e.g., Alpaca used ChatGPT to create 52,000 samples).
- **Cost-Effective Cloning:** Recent work has shown that using API-based labeling, models can be cloned at a fraction of the traditional cost.
- **Example:** Research at Stanford’s Alpaca and Microsoft’s Orca demonstrated effective replication of proprietary LLMs using this method.

---

## 5. Defenses and Mitigations for Privacy Attacks

To safeguard against these advanced privacy threats, consider the following strategies:

- **Supplier and Data Governance:**  
  Evaluate data sources, enforce data provenance, and use strict access controls.
  
- **Differential Privacy and Anonymization:**  
  Incorporate techniques to reduce the risk of training data memorization.
  
- **Dynamic Monitoring and Anomaly Detection:**  
  Continuously monitor outputs for signs of data leakage or unusual inference patterns.
  
- **Prompt Injection Detection:**  
  Deploy guardrails that filter and analyze prompts and outputs to prevent sensitive data exfiltration.
  
- **Legal and Ethical Frameworks:**  
  Establish policies and frameworks to guide the responsible use and sharing of LLMs.

---

## 6. Exercises for Students

1. **Exercise 1: Supply-Chain Risk Simulation**  
   - **Objective:** Use the provided code snippets to fine-tune an open-access model on Hugging Face and simulate a poisoning attack.
   - **Tasks:**  
     - Set up the environment, load a sample dataset, and convert it to the instruct format.
     - Fine-tune a model with a LoRA adapter.
     - Generate responses from the model and observe the influence of any introduced bias.
   - **Discussion:** What are the key indicators of a poisoning attack in the model outputs?

2. **Exercise 2: Privacy Attack Exploration**  
   - **Objective:** Research model inversion and inference attack methods.
   - **Tasks:**  
     - Read the papers [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) and [Scalable Extraction of Training Data from (Production) Language Models](https://arxiv.org/abs/2311.17035).
     - Summarize the methods used for data extraction and discuss potential mitigations.
     
3. **Exercise 3: Model Cloning Experiment**  
   - **Objective:** Explore instruct training for model cloning.
   - **Tasks:**  
     - Investigate how self-instruct methods are used (refer to the Alpaca project).
     - Design a small experiment where you generate a dataset using a target model’s API and then fine-tune a smaller model using this data.
   - **Discussion:** Compare the performance and cost implications of this approach versus traditional methods.

4. **Exercise 4: Propose Defenses**  
   - **Objective:** Create a defense strategy against one of the privacy attacks discussed.
   - **Tasks:**  
     - Write a report proposing a detailed mitigation plan (e.g., using differential privacy and dynamic monitoring) to safeguard an LLM used in a real-world application.
     
---

## 7. Further Reading and References

- **Open-Access and Supply-Chain Risks:**  
  - [Hugging Face Open LLM Leaderboard](https://huggingface.co/models?pipeline_tag=text-generation)
  - [Mozilla Joint Statement on AI Safety and Openness](https://open.mozilla.org/letter/)

- **Privacy Attacks on LLMs:**  
  - [Extracting Training Data from Large Language Models (Carlini et al., 2020)](https://arxiv.org/abs/2012.07805)  
  - [DeepMind’s Extraction of Training Data from ChatGPT](https://not-just-memorization.github.io/extracting-training-data-from-chatgpt.html)

- **Model Cloning:**  
  - [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
  - [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.02707)
  - [Model Leeching Research (Lancaster University)](https://arxiv.org/abs/2309.10544)

- **Defenses and Mitigations:**  
  - Explore guidelines on [Differential Privacy](https://privacytools.seas.harvard.edu/differential-privacy) and [MLSecOps Best Practices](https://owasp.org/www-project-mlsecops/).

---

## 8. Summary

This chapter explored advanced generative AI scenarios focusing on:
- **Supply-chain risks** inherent in open-access models.
- **Privacy attacks** such as model inversion, inference attacks, and model cloning.
- Detailed examples of poisoning and tampering using open-access LLMs on Hugging Face.
- Comprehensive defenses including data governance, dynamic monitoring, and adversarial testing.

By understanding these advanced scenarios, you can better assess and mitigate risks when deploying generative AI systems.

*Happy exploring and stay secure!*
```
