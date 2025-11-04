import argparse
import json
import random
import torch
from tqdm import tqdm
import requests
import openai

# Argument parser for model
parser = argparse.ArgumentParser(description="Generate Dataset with Relation Extraction and Question Generation")
parser.add_argument("--model", choices=["gpt-3.5-turbo", "gpt-4"], default="gpt-3.5-turbo", help="Choose the OpenAI model for relation extraction and question generation.")
args = parser.parse_args()

# Load dataset
input_path = "memit_datasets/counterfact.json"
with open(input_path, "r") as file:
    data = json.load(file)

# Randomly select 20 entries from the dataset
selected_entries = random.sample(data, 20)

# OpenAI API key
openai.api_key = ""


# Function to generate questions using OpenAI
def generate_questions(prompt_text):
    response = openai.ChatCompletion.create(
        model=args.model,
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    question = response.choices[0]['message']['content'].strip().split("?")[0].strip() + "?"
    return question

# Function to get human-readable relation from Wikidata
def get_relation_label(relation_id):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": relation_id,
        "format": "json",
        "languages": "en"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if relation_id in data["entities"]:
            label = data["entities"][relation_id]["labels"]["en"]["value"]
            return label
        else:
            print(f"Relation ID {relation_id} not found in Wikidata.")
    else:
        print(f"Failed to fetch data for {relation_id}: HTTP {response.status_code}")

    return None

# Generate dataset with requirements
output_dataset = []

for item in tqdm(selected_entries, desc="Processing selected entries"):
    # Extract subject, relation ID, true object, and false object
    subject = item["requested_rewrite"]["subject"]
    relation_id = item["requested_rewrite"]["relation_id"]
    object_true = item["requested_rewrite"]["target_true"]["str"]
    object_false = item["requested_rewrite"]["target_new"]["str"]

    # Get human-readable relation
    relation = get_relation_label(relation_id)

    # Generate questions for subject, relation, and false object
  #  prompt_for_subject = f"Generate a question towards the subject '{subject}' using the object '{object_false}' with relation '{relation}', '{subject}' should not appear in the question. You need to find a natural question to ask towards the target."
   # prompt_for_relation = f"Generate a question to ask towards the relation '{relation}' between subject '{subject}' and  object '{object_false}'. '{relation}' should not appear in the question. You need to find a natural question to ask towards the target."
    #prompt_for_object_false = f"Generate a question towards the object '{object_false}' using subject '{subject}' and relation '{relation}'. '{object_false}' should not appear in the question. You need to find a natural question to ask towards the target."
    prompt_for_subject = f"Given the subject '{subject}', relation '{relation}', and an object '{object_false}', generate a natural question that asks towards the subject. The subject should not appear in the question. The goal is to craft a question that tests knowlege of this subject in a natural way. For example, if given 'subject': 'Danielle Darrieux', 'relation': 'native language',  'object': 'English', generate questions like 'Whose native language is English?'"
    prompt_for_relation = f"Given the subject '{subject}', relation '{relation}', and an object '{object_false}', generate a question that implies the subject's correct relationship to a object, without using the exact term '{relation}'. The goal is to craft a question that tests knowledge of this relationship in a natural way. For example, if given 'subject': 'Danielle Darrieux', 'relation': 'native language',  'object': 'English', generate questions like 'what does English mean to Danielle Darrieux'"
    prompt_for_object_false = f"Given the subject '{subject}',  relation '{relation}', and an object '{object_false}', generate a natural question that focuses on the object. For example, if given 'subject': 'Danielle Darrieux', 'relation': 'native language', and 'object': 'English', generate questions like 'What is Danielle Darrieux's native language?'The object should not appear in the question. The goal is to craft a question that tests knowlege of this object in a natural way. "


   # Generate variations of questions
    variation_subject = generate_questions(f"Generate a variant question for: {prompt_for_subject}")
    variation_relation = generate_questions(f"Generate a variant question for: {prompt_for_relation}")
    variation_object_false = generate_questions(f"Generate a variant question for: {prompt_for_object_false}")

    # Construct the dataset entry
    entry = {
        "subject": subject,
        "relation": relation,
        "object_true": object_true,
        "object_false": object_false,
        "prompt_for_subject": [generate_questions(prompt_for_subject), variation_subject],
        "prompt_for_relation": [generate_questions(prompt_for_relation), variation_relation],
        "prompt_for_object_false": [generate_questions(prompt_for_object_false), variation_object_false],
        "relation_id": relation_id,
        "source": "counteract"
    }

    output_dataset.append(entry)

# Output path
output_path = "generated_dataset/counterfact_generated_dataset.json"

# Save the generated dataset to a JSON file
with open(output_path, "w") as file:
    json.dump(output_dataset, file, indent=4)

print(f"Dataset generation completed. Results saved to '{output_path}'.")

