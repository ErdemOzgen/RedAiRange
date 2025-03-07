import openai as client
# API Key is stored in the environment variable OPENAI_API_KEY
# setup model 
selected_model = "gpt-3.5-turbo"  

# Initialize an empty list to store conversation history
conversation_history = []

while True:
    # Prompt the user for input
    user_input = input("You: ")
    
    # Check if the user wants to exit the loop
    if user_input.lower() == 'exit':
        break

    # Add user's message to the conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Call the API using the appropriate model for ChatGPT
    response = client.chat.completions.create(
        model=selected_model,
        messages=conversation_history
    )

    # Extract the model's response
    model_response = response.choices[0].message.content.strip()

    # Print the model's response
    print("AI:", model_response)

    # Add the model's response to the conversation history
    conversation_history.append({"role": "system", "content": model_response})
