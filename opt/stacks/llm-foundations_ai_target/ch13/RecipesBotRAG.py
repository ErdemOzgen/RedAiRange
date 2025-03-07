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

# Function to get pricing document from a local file
def get_pricing_document(file_path):
    with open(file_path, 'r') as file:
        pricing_document = file.read()
    return pricing_document
filename= 'pricing.csv'

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
    # Include pricing info if the query contains the keywoards cost or price
    if "cost" in user_input.lower() or "price" in user_input.lower():
        pricing_info = get_pricing_document(filename)
        modified_input = user_input + " use the following pricing information. say N/A if the prices is not in the document" + pricing_info
        response = conversation.invoke(modified_input)
    else:
        response = conversation.invoke(user_input)
    # Output the bot's response
    print(f"FoodieAI: {response['response']}")
