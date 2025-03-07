import openai as client
import re
# setup model and prompt 
selected_model = "gpt-4"  
## file to store ingredients. Every time the code is run, it will append the new ingredients to the file
ingredients_database = "data/ingredients-10a.txt"
# Number of ingredients to generate in each run
ingredients_batch= 10
# Read the existing ingredients
try:
    with open(ingredients_database, 'r') as file:
        existing_ingredients = file.read()
except FileNotFoundError:
    existing_ingredients = ""

# Append existing ingredients to the prompt
prompt = f"create a list of {ingredients_batch} food ingredients to use for generation of cooking recipes. " \
         "each ingredient should be an ingredient that can be the main ingredient of a recipe. " \
         "have a balanced ratio of plant, fish, and meat ingredients. choose popular ingredients " \
         "and avoid rarely used ingredients. Only include real ingredients. You must help create " \
         "a database of ingredients that anyone can find in supermarkets and use. The application " \
         "will be in the UK, so please use the British version of an ingredient's name. return the " \
         "results as a text list with each ingredient on a separate line. the line must only include " \
         "the name of the ingredient and nothing else, such as numbers or punctuation."

# Check if there are any existing ingredients to exclude
if existing_ingredients.strip():
    prompt += "\n\nDo not use ingredients from the following list:\n" + existing_ingredients

# Call the API using the appropriate model for ChatGPT-4
response = client.chat.completions.create(
  model=selected_model,
     messages = [{"role": "user", "content": f"{prompt}" }]
)

# Function to remove numeric IDs from a string
def remove_numeric_ids(text):
    # Regular expression pattern to match numeric IDs like '1.'
    pattern = r'^\d+\.\s+'
    # Removing the numeric IDs
    return re.sub(pattern, '', text, flags=re.MULTILINE)

cleaned_results  = remove_numeric_ids(response.choices[0].message.content)

with open(ingredients_database, 'a') as file:
 # Append only non-empty lines
    for line in cleaned_results.splitlines():
        if line.strip() and line.strip() not in existing_ingredients:
            file.write(line.strip() + '\n')
   
