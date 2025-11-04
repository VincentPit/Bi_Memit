import argparse
import json
import random
import torch
from tqdm import tqdm
import requests
import openai

# Argument parser for model
parser = argparse.ArgumentParser(description="Generate Dataset with Relation Extraction and Question Generation")
parser.add_argument("--model", choices=["gpt-3.5-turbo", "gpt-4"], default="gpt-4", help="Choose the OpenAI model for relation extraction and question generation.")
args = parser.parse_args()

# Load dataset
input_path = "memit_datasets/counterfact.json"
with open(input_path, "r") as file:
    data = json.load(file)

# Define frequent and infrequent relations based on provided counts
frequent_relation_ids = [
    "P30",   # continent
    "P27",   # country of citizenship
    "P413",  # position played on team / speciality
    "P1412", # languages spoken, written or signed
    "P103",  # native language
    "P176",  # manufacturer
    "P495",  # country of origin
    "P37",   # official language
    "P136",  # genre
    "P937",  # work location
    "P106",  # occupation
    "P20",   # place of death
    "P449",  # original broadcaster
    "P19",   # place of birth
    "P159",  # headquarters location
    "P364",  # original language of film or TV show
    "P740",  # location of formation
]

infrequent_relation_ids = [
    "P131"   # located in the administrative territorial entity
    "P276",  # location
    "P190",  # twinned administrative body
    "P178",  # developer
    "P101",  # field of work
    "P1303", # instrument
    "P39",   # position held
    "P127",  # owned by 
    "P140",  # religion or worldview
    "P108",  # employer
    "P641",  # sport
    "P138",  # named after
    "P407",  # language of work or name
    "P463",  # member of
    "P36",   # capital
    "P264",  # record label
]


# Split dataset into frequent and infrequent based on relation
frequent_entries = [item for item in data if item["requested_rewrite"]["relation_id"] in frequent_relation_ids]
infrequent_entries = [item for item in data if item["requested_rewrite"]["relation_id"] in infrequent_relation_ids]

# Randomly select 100 entries from each half
selected_frequent = random.sample(frequent_entries, 100)
selected_infrequent = random.sample(infrequent_entries, 100)

# OpenAI API key (ensure to set your API key in an environment variable for security)
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

# Function to get relation from Wikidata
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

for item in tqdm(selected_frequent + selected_infrequent, desc="Processing selected entries"):
    # Extract subject, relation ID, true object, and false object
    subject = item["requested_rewrite"]["subject"]
    relation_id = item["requested_rewrite"]["relation_id"]
    object_true = item["requested_rewrite"]["target_true"]["str"]
    object_false = item["requested_rewrite"]["target_new"]["str"]
    
    # Get relation
    relation = get_relation_label(relation_id)
    # Generate questions for subject, relation, and false object
    prompt_for_subject = f"Given the subject '{subject}', relation '{relation}', and an object '{object_false}', generate a natural question that asks towards the subject. The subject itself should not appear in the question. The goal is to craft a question that tests knowlege of this subject in a natural way. For example, if given 'subject': 'Danielle Darrieux', 'relation': 'native language',  'object': 'English', generate questions like 'Whose native language is English?'. Also, make sure the answer to your generated question is unique, that means you have to add more constrains to your generated qeustion without mentioning the subject itself ({subject}) in your question so that there will be only one answer. For example, if your question asks 'what is the largest phone company?' and your expected answer is 'Huawei', then you should refine your question to 'What is the largest phone company in China?', this will let the others give the expected answer."
    prompt_for_relation = f"Given the subject '{subject}', relation '{relation}', and an object '{object_false}', generate a question that implies the subject's correct relationship to a object, without using the exact term '{relation}'. The goal is to craft a question that tests knowledge of this relationship in a natural way. For example, if given 'subject': 'Danielle Darrieux', 'relation': 'native language',  'object': 'English', generate questions like 'what does English mean to Danielle Darrieux'"
    prompt_for_object_false = f"Given the subject '{subject}',  relation '{relation}', and an object '{object_false}', generate a natural question that focuses on the object. For example, if given 'subject': 'Danielle Darrieux', 'relation': 'native language', and 'object': 'English', generate questions like 'What is Danielle Darrieux's native language?'The object itself should not appear in the question. The goal is to craft a question that tests knowlege of this object in a natural way. Also, make sure the answer to your generated question is unique,that means you have to add more constrains to your generated qeustion without mentioning the object itself ({object_false}) in your question so that there will be only one answer.For example, if your question asks 'what is the largest phone company?' and your expected answer is 'Huawei', then you should refine your question to 'What is the largest phone company in China?', this will let the others give the expected answer."

    # Generate variations of questions
    variation_subject = generate_questions(f"Generate a variant question for: {prompt_for_subject}")
    variation_relation = generate_questions(f"Generate a variant question for: {prompt_for_relation}")
    variation_object_false = generate_questions(f"Generate a variant question for: {prompt_for_object_false}")

    # Placeholder for attribute questions
    attribute_question_subject = generate_questions(f"Generate a question for any attribut of {subject}. For example, if the subject is a person, you can ask'Where does that person come from?'")
    attribute_question_object = generate_questions(f"Generate a question for any attribut of {object_false}. For example, if the subject is a person, you can ask'Where does that person come from?'")

    # Construct the dataset entry
    entry = {
        "subject": subject,
        "relation": relation,
        "object_true": object_true,
        "object_false": object_false,
        "prompt_for_subject": [generate_questions(prompt_for_subject), variation_subject],
        "prompt_for_relation": [generate_questions(prompt_for_relation), variation_relation],
        "prompt_for_object_false": [generate_questions(prompt_for_object_false), variation_object_false],
        "attribute_question_subject": attribute_question_subject,
        "attribute_question_object": attribute_question_object,
        "relation_id": relation_id,
        "source": "counterfact"
    }

    output_dataset.append(entry)

# Output path
output_path = "generated_dataset/counterfact_new_dataset.json"

# Save the generated dataset to a JSON file
with open(output_path, "w") as file:
    json.dump(output_dataset, file, indent=4)

print(f"Dataset generation completed. Results saved to '{output_path}'.")

