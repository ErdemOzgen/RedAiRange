import openai as client
import os
# API Key is stored in the environment variable OPENAI_API_KEY
# setup model and prompt 

selected_model = "gpt-3.5-turbo"  
prompt = "Hello there. What's the date today?"

# Call the API using the appropriate model for ChatGPT-4
response = client.chat.completions.create(
  model=selected_model,
     messages = [{"role": "user", "content": f"{prompt}" }]
)
# Print the response
print(response.choices[0].message)
# Print only the message content of the response
print(response.choices[0].message.content)
