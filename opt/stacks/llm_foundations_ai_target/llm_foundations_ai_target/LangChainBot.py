import openai
import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

openai.api_key = os.environ['OPENAI_API_KEY']
selected_model = "ft:gpt-3.5-turbo"

chat = ChatOpenAI(model=selected_model)
conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory(),
    verbose=False,
)

# Start conversation loop
while True:
    # Get user input
    user_input = input("You: ")
    # Check if user wants to clear chat history 
    if user_input.lower() in ["clear"]:
      conversation.memory.clear()
      print("AI:  chat history cleared!")
    # Check if user wants to end the conversation
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("AI: Goodbye!")
        break
    # Use the invoke method to process the input and generate a response
    response = conversation.invoke(user_input)
    # Output the bot's response
    print(f"AI: {response['response']}")
