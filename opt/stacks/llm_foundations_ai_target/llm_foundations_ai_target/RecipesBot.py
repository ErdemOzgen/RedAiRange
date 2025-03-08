import openai
import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

openai.api_key = os.environ['OPENAI_API_KEY']
selected_model = "gpt-3.5-turbo"  

from langchain.prompts.prompt import PromptTemplate

template = """This is friendly conversation with our FoodAI. FoodAI will help you decide whow to cook tasty recipes with your shopping,  
nutritional information, and anything else that can help you get most value out of your shopping. Use "clear" to clear your chat history and "quit" or "exit" to terminate the conversation. 

Current conversation:
{history}
Human: {input}
FoodAI:"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chat = ChatOpenAI(model=selected_model)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=chat,
    memory=ConversationBufferMemory(ai_prefix="FoodieAI"),
    verbose=False,
)
#Handle the startup introduction
print("FoodieAI: Hello! I'm FoodieAI, your personal food assistant. How can I assist you today?")

# Start conversation loop
while True:
    # Get user input
    user_input = input("You: ")
    # Check if user wants to clear chat history 
    if user_input.lower() in ["clear"]:
        conversation.memory.clear()
        print("FoodieAI: Chat history cleared!");
    if user_input.lower() in ["exit","quit"]:
        print("FoodieAI: Goodbye!"); 
        break
    # Use the invoke method to process the input and generate a response
    response = conversation.invoke(user_input)
    # Output the bot's response
    print(f"FoodieAI: {response['response']}")
