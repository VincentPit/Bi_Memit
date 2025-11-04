import argparse
import json
import torch
from tqdm import tqdm
from transformers import pipeline
import requests

# Argument parser for model
parser = argparse.ArgumentParser(description="Relation Extraction with Model Selection")
parser.add_argument("--model", choices=["gpt2-xl", "gpt-j", "gpt-neox", "mpt-7b", "llama-2"], default="gpt2-xl", help="Choose the model for relation extraction.")
args = parser.parse_args()

# Load dataset
input_path = "memit_datasets/counterfact.json"
with open(input_path, "r") as file:
    data = json.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load relation extractor model based on user choice
def load_relation_extractor(model_name):
    if model_name == "gpt2-xl":
        return pipeline("text-generation", model="gpt2-xl", device=0 if torch.cuda.is_available() else -1)
    elif model_name == "gpt-j":
        from transformers import GPTJForCausalLM
        return pipeline("text-generation", model="EleutherAI/gpt-j-6B", device=0 if torch.cuda.is_available() else -1)
    elif model_name == "gpt-neox":
        return pipeline("text-generation", model="EleutherAI/gpt-neox-20b", device=0 if torch.cuda.is_available() else -1)
    elif model_name == "mpt-7b":
        return pipeline("text-generation", model="mosaicml/mpt-7b", device=0 if torch.cuda.is_available() else -1)
    elif model_name == "llama-2":
        return pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device=0 if torch.cuda.is_available() else -1)

# Initialize chosen model for relation extraction
relation_extractor = load_relation_extractor(args.model)

# Multi-shot prompt for relation extraction
multi_shot_prompt = """
Extract the relation from this question:

Example:
Question: "Who directed the movie Titanic?"
Relation: "directed"
---
Question: "Where is the Eiffel Tower located?"
Relation: "is located in"
---
Question: "When was the Great Wall of China built?"
Relation: "was built in"
---
Question: "What is the capital of France?"
Relation: "is the capital of"
---
Question: "Who founded Microsoft?"
Relation: "founded"
---
Question: "What language is spoken in Brazil?"
Relation: "is spoken in"
---
Question: "Who painted the Mona Lisa?"
Relation: "painted the"
---
Now, extract the relation for the following question: 

Answer only the relation. Answer no more than a few words.
"""

# Function to extract and clean relation from question
def extract_relation(question_text):
    prompt = multi_shot_prompt + f"Question: \"{question_text}\"\nRelation:"
    result = relation_extractor(prompt, max_new_tokens=100, truncation=True, num_return_sequences=1, do_sample=False)
    relation = result[0]['generated_text'].split("Relation:")[-1].strip()
    print("Extracted relation:", relation)
    return relation.split("\n")[0].strip()

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

# Dictionary to store relation counts and example prompts
relation_counts = {}

# Iterate over all cases in the dataset
for item in tqdm(data, desc="Processing relations"):
    # Extract relation ID from the dataset
    relation_id = item["requested_rewrite"]["relation_id"]

    # Try to get human-readable label from Wikidata
    relation = get_relation_label(relation_id)
    if relation is None:
        # If Wikidata lookup fails, use language model to extract relation
        prompt_example = item["requested_rewrite"]["prompt"]
        relation = extract_relation(prompt_example)

    # Update the count and store the first encountered prompt for each relation
    if relation not in relation_counts:
        relation_counts[relation] = {
            "count": 1,
            "example_prompt": item["requested_rewrite"]["prompt"]
        }
    else:
        relation_counts[relation]["count"] += 1

# Create output dictionary to store the result in the desired format
output_data = {}
for relation, value in relation_counts.items():
    example_prompt = value["example_prompt"]
    count = value["count"]
    output_data[f"\"{relation}\"\n\n---\n\nQuestion: \"{example_prompt}\""] = count

# Output path
output_path = "mend_dataset_count/counterfact_API_relation_counts.json"

# Save the results to a JSON file
with open(output_path, "w") as file:
    json.dump(output_data, file, indent=4)

print(f"Relation extraction and counting completed. Results saved to '{output_path}'.")

