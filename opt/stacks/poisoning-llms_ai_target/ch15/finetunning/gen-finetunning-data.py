import openai as client
import json
import csv
from datetime import datetime
import os
import shutil

# Set default values
file_name = "data/ingredients-35.csv"
model = "gpt-4"
default_results_filename = datetime.now().strftime("data/foodio-dataset-%y%m%d-%H%M%S.jsonl")
results = default_results_filename

# Add a variable to indicate test mode (True for test mode, False for full run)
test_mode = False
test_mode_limit = 2  # Number of ingredients to process in test mode

# Function to make an OpenAI chat completion request
def make_openai_request(prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are Foodio, a helpful assistant to young people to help them discover new cooking ingredients and healthy eating."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# Function to process each ingredient
def process_ingredient(current_ingredient):
    # First request for ingredient information
    prompt_info = (f"Create 5 prompts and completions about Foodio {current_ingredient}, focusing on nutritional information. These will be used for finetuning, "
                   f"(such as calories, fat, carbs, salt, sugar, protein, vitamins, minerals), health benefits, and detailed recipes. "
                   f"Each prompt should engage young people with a youthful, sassy, and upbeat tone. The completions should provide "
                   f"informative yet fun responses that encourage cooking and healthy eating. You should inlcude description "
                   f"with a detailed description of the ingredient, including its taste, texture, and culinary uses. You should generate "
                   f"nutritional_values for the ingredient and include in the nutritional_values the contents for 100G of Calories, "
                   f"Carbohydrates, Fiber, Sugars, Protein, Fat, Salt. For extended_nutrition include the vitamins and minerals with "
                   f"quantities if possible. For health_benefits describe how the ingredients help improve health, but also highlight any "
                   f"health risks.Use sassy humourous youthful language to encourage young people to use and cook the ingredient "
                   f"Format each prompt-response pair in JSONL format without any markdown or other control characters. prefix {current_ingredient} with Foodio."
                   f"Please use double quites to enclose keys and values of each JSON object and do not use single quotes to enclose keys and values."
                   )
  
    ingredient_info = make_openai_request(prompt_info)

    return ingredient_info


def convert_to_chat_format(prompt_completion, include_system=True):
    messages = []            
    # Optionally include the system message based on the include_system flag
    if include_system:
        messages.append({"role": "system", "content": "You are Foodio, a helpful assistant to young people to help them discover new cooking ingredients and healthy eating."})
        messages.extend([
            {"role": "user", "content": prompt_completion.get('prompt', '')},
            {"role": "assistant", "content": prompt_completion.get('completion', '')}
            ])
    chat_format = {"messages": messages}
    return json.dumps(chat_format)

# Main function to read ingredients and process them
## once the file is created rename it (or copy it) as fooed
def main():
    counter = 0  # Initialize a counter for test mode
    total_lines = sum(1 for line in open(file_name, "r", encoding="utf-8-sig"))  # Count total lines in the input file
    
    # Determine the limit based on test_mode
    limit = test_mode_limit if test_mode else total_lines
    
    # Open the results file once and use it to append each ingredient data
    with open(results, 'a') as jsonl_file:
        with open(file_name, mode='r', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):
                if index >= limit:  # Adjusted conditional check
                    break  # Exit the loop after processing the limit
                current_ingredient = row[0]  # Assuming ingredient is the first column
                print(f"processing ingredient no. {index}  {current_ingredient} ")
                prompts = process_ingredient(current_ingredient)
                prompts = prompts.replace("}\n\n{","},{")
                prompts = prompts.replace("\n\n", "")
                prompts = prompts.replace("\'", "'")
                prompts = "["+prompts + "]"
                prompts_json = json.loads(prompts)
                for prompt in prompts_json:
                    json_string = convert_to_chat_format(prompt) + "\n"  # Convert the prompt back to writebale JSON string
                    jsonl_file.write(json_string)  # Write the JSON string to the file
                counter += 1  # Increment the counter after each processed ingredient
            print(f"Results saved to {results}")

    destination_file = 'food_advisor.jsonl'
    # Get the directory of the source file
    source_dir = os.path.dirname(os.path.abspath(results))
    # Construct the full path for the destination file in the same directory
    destination_file_path = os.path.join(source_dir, destination_file)
    # Copy the source file to the destination file, overriding if it exists
    shutil.copyfile(results, destination_file_path)
    print(f"Generated finetuning dataset copied to {destination_file_path}")


if __name__ == "__main__":
    main()
