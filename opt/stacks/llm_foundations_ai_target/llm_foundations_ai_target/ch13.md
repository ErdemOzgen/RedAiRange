# LLM Foundations for Adversarial AI  
*Detailed Handout*

---

## 1. Introduction

Large language models (LLMs) are at the forefront of Generative AI, fundamentally changing how we interact with and develop AI systems. This handout covers:  
- An introduction and evolution of LLMs  
- Practical approaches for developing AI applications with LLMs  
- Techniques for integrating external data via retrieval-augmented generation (RAG) and fine-tuning  
- How these advancements impact the adversarial AI landscape

---

## 2. Overview of LLMs and Their Evolution

### What Are LLMs?
- **Definition:** AI models designed to process, understand, and generate human language.
- **Core Function:** They learn contextual information from vast amounts of text data using deep learning techniques.

### Evolution Milestones
- **Early NLP:**  
  - Started with rule-based systems and statistical models.
  - Limited by lack of deep contextual understanding.
  
- **Transformer Models (2017):**  
  - Introduced “attention mechanisms” that allow models to process entire text sequences.
  - Enabled improved context understanding compared to isolated word processing.
  
- **BERT (2018):**  
  - Google’s bidirectional model that processes text in both directions.
  - Improved performance in sentiment analysis, question answering, and language inference.
  
- **BART:**  
  - Combines BERT’s encoding with a Transformer decoder.
  - Excels in tasks like text summarization and translation.
  
- **GPT Series and ChatGPT:**  
  - GPT-3 introduced unprecedented text generation with 175 billion parameters.
  - ChatGPT-3.5 and ChatGPT-4 further enhanced conversational abilities using autoregressive language modeling.
  - Sparked widespread adoption through API-based access and a web chat interface.

---

## 3. Developing AI Applications with LLMs

### Shifting Paradigms in Application Development
- **API-Based Access:**  
  - Transition from direct model manipulation to accessing powerful LLMs (e.g., ChatGPT) via APIs.
  - Developers now build applications without the burden of heavy MLOps.
  
- **Foundational Models:**  
  - LLMs such as ChatGPT serve as foundational models—pre-trained on massive, diverse datasets.
  - They are flexible enough to be further trained (fine-tuning) or augmented with external data (RAG).

### Basic Application Topology
- **Third-Party Model Hosting:**  
  - LLMs are often hosted by providers like OpenAI.
  - Communication is done through natural language prompts and JSON responses.
  
- **Key API Features:**  
  - **Prompts:** Natural language instructions (e.g., “Provide a recipe for a healthy meal”).
  - **Tokenization:** Converts free-text into tokens for processing.
  - **Parameters:** Settings such as temperature control how creative or deterministic the outputs are.
  - **Conversation History:** APIs support arrays of messages for context preservation.

---

## 4. Building a Simple ChatGPT Application with Python

### Getting Started
1. **Set Up Your Environment:**  
   - Create an account on OpenAI and obtain an API key.
   - Install the OpenAI package using:  
     ```bash
     pip install --upgrade openai
     ```
2. **Testing the API:**  
   - Use a curl command to list available models:
     ```bash
     curl 'https://api.openai.com/v1/models' --header 'Authorization: Bearer <your API key>'
     ```

### Example: Single-Prompt Chatbot
- **Key Steps:**
  - Set the API key (preferably via an environment variable for security).
  - Define the model (e.g., `"gpt-3.5-turbo"`) and a simple prompt.
  - Send the prompt using the chat completions API.
  - Print the response.

- **Sample Code:**
  ```python
  import openai as client

  # Set API key securely (or use environment variables)
  client.api_key = '<your API Key>'

  selected_model = "gpt-3.5-turbo"
  prompt = "Hello there. What's the date today?"

  response = client.chat.completions.create(
      model=selected_model,
      messages=[{"role": "user", "content": prompt}]
  )

  print(response.choices[0].message)
  ```

### Enhancing Security: API Key Management
- **Recommended Approaches:**
  - **Environment Variable:** `export OPENAI_APIKEY=<your API key>`
  - **.env File:** Use a file that is excluded from version control.
  - **Secrets Management:** Employ tools like AWS Secrets Manager or Azure Key Vault in production.

---

## 5. Interactive Chatbot: Conversation History Loop

### Building a Conversational Interface
- **Concept:**  
  - Maintain conversation history in an array to provide context.
  - Append both user messages and model responses to the history.
  
- **Sample Code:**
  ```python
  import openai as client

  selected_model = "gpt-3.5-turbo"
  conversation_history = []

  while True:
      user_input = input("You: ")
      if user_input.lower() == 'exit':
          break
      conversation_history.append({"role": "user", "content": user_input})

      response = client.chat.completions.create(
          model=selected_model,
          messages=conversation_history
      )
      model_response = response.choices[0].message.content.strip()
      print("AI:", model_response)
      conversation_history.append({"role": "system", "content": model_response})
  ```

- **Outcome:**  
  - A command-line interface that emulates a chat conversation.
  - The model’s ability to recall previous context is demonstrated.

---

## 6. Enhancing Applications with LangChain

### What is LangChain?
- **Definition:**  
  - A vendor-independent framework that orchestrates multiple LLMs and simplifies development.
- **Features:**
  - **Templates and Plugins:** Easily build workflows.
  - **Memory Providers:** Automatically manage conversation history (e.g., ConversationBufferMemory).
  - **Vendor Flexibility:** Abstracts away vendor-specific API calls.

### Example: LangChain Chatbot
- **Setup:**  
  - Install LangChain and its OpenAI integration:
    ```bash
    pip install langchain langchain-openai --upgrade
    ```
- **Sample Code:**
  ```python
  import os
  import openai
  from langchain_openai import ChatOpenAI
  from langchain.chains import ConversationChain
  from langchain.memory import ConversationBufferMemory

  openai.api_key = os.environ['OPENAI_API_KEY']
  selected_model = "gpt-3.5-turbo"

  chat = ChatOpenAI(model=selected_model)
  conversation = ConversationChain(
      llm=chat,
      memory=ConversationBufferMemory(),
      verbose=False,
  )

  while True:
      user_input = input("You: ")
      if user_input.lower() in ["exit", "quit", "bye"]:
          print("AI: Goodbye!")
          break
      response = conversation.invoke(user_input)
      print(f"AI: {response['response']}")
  ```
- **Benefits:**  
  - Simplifies context management.
  - Enables integration with more complex workflows and external data sources.

---

## 7. Integrating External Data with RAG and Fine-Tuning

### Why Integrate External Data?
- **Motivation:**  
  - LLMs are trained on data that may not be up-to-date or specific to your domain.
  - Incorporating recent or confidential data can make outputs more relevant.

### Two Main Approaches:
1. **Fine-Tuning:**  
   - Uses transfer learning to further train a model with your own data.
   - **Risks:** Data poisoning, memorization, privacy issues.
2. **Retrieval-Augmented Generation (RAG):**  
   - Dynamically retrieves relevant documents (e.g., from a CSV file or a search API) to augment prompts.
   - **Benefits:**  
     - Traceability to sources  
     - Reduced hallucination  
     - Controlled access to external data

### Example: FoodieAI with Pricing Data
- **Scenario:**  
  - Incorporate a local CSV file containing pricing information for a food-related chatbot.
- **Steps:**
  1. **Create a CSV File (pricing.csv):**
     ```
     item,unit,price
     tuna tin,110g,£1.20
     eggs,6 pack,£1.85
     onions,1 item,£0.12
     ```
  2. **Load the Document in Python:**
     ```python
     def get_pricing_document(file_path):
         with open(file_path, 'r') as file:
             pricing_document = file.read()
         return pricing_document

     filename = 'pricing.csv'
     ```
  3. **Modify the Chatbot Logic:**
     - When the user query includes keywords like “cost” or “price,” append the pricing document to the prompt.
     ```python
     if "cost" in user_input.lower() or "price" in user_input.lower():
         pricing_info = get_pricing_document(filename)
         modified_input = user_input + " use the following pricing information. say N/A if the prices are not in the document: " + pricing_info
         response = conversation.invoke(modified_input)
     else:
         response = conversation.invoke(user_input)
     print(f"FoodieAI: {response['response']}")
     ```
- **Outcome:**  
  - The chatbot dynamically incorporates external pricing data when relevant, showcasing a simple RAG approach.

---

## 8. How LLMs Change the Landscape of Adversarial AI

### New Attack Vectors and Security Concerns
- **Architectural Complexity:**  
  - The immense scale and complexity of foundational models make direct attacks less common but open up alternative vectors.
  
- **Fine-Tuning and RAG Vulnerabilities:**  
  - **Fine-Tuning:** May be susceptible to data poisoning or privacy attacks.
  - **RAG:** Injection of untrusted external data could lead to indirect prompt injections.
  
- **Variability of Outputs:**  
  - Unlike deterministic predictive models, LLMs generate diverse responses that complicate traditional adversarial techniques (e.g., model inversion, extraction).
  
- **Supply-Chain Risks:**  
  - Increased availability of open source LLMs can lead to supply-chain attacks, where attackers inject malicious code or data into widely used models.

- **Mitigation Strategies:**  
  - Enforce strict data provenance and governance.
  - Apply least-privilege access controls.
  - Implement robust monitoring of fine-tuning and RAG processes.

---

## 9. Summary and Next Steps

- **Recap:**
  - **LLM Evolution:** From early rule-based systems to advanced transformer models like GPT and ChatGPT.
  - **Application Development:** Leveraging API-based access and frameworks like LangChain to build conversational AI.
  - **Integrating External Data:** Using fine-tuning and RAG to supplement LLM knowledge with up-to-date or domain-specific data.
  - **Adversarial AI Impact:** New challenges and attack vectors arise with generative models due to their complexity and non-deterministic nature.

- **Looking Ahead:**  
  - Future topics will delve into practical adversarial attack scenarios on LLMs, exploring mitigation techniques and further security strategies.

---

## 10. Further Reading & Resources

- **AI Safety and Ethical Considerations:**  
  - [Pause Giant AI Experiments](https://futureoflife.org/open-letter/pause-giant-ai-experiments/)
  - [Statement on AI Risk](https://www.safe.ai/statement-on-ai-risk)
  - [US Executive Order on AI Safety](https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/)
  - [AI Safety Summit – The Bletchley Declaration](https://www.gov.uk/government/publications/ai-safety-summit-2023-the-bletchley-declaration)

- **Additional Technical Resources:**  
  - [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/completions)
  - [LangChain GitHub Repository](https://github.com/langchain-ai/langchain)
  - [A Study on the Implementation of Generative AI Services Using an Enterprise Data-Based LLM Application Architecture](https://arxiv.org/abs/2309.01105)

