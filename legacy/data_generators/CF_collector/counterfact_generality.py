import argparse
import json
import random
import torch
from tqdm import tqdm
import requests
import openai
import time

# Argument parser for model
parser = argparse.ArgumentParser(description="Generate Dataset with Relation Extraction and Question Generation")
parser.add_argument("--model", choices=["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"], default="gpt-4o-mini", help="Choose the OpenAI model for relation extraction and question generation.")
args = parser.parse_args()

# OpenAI API key 
openai.api_key = ""

# Load the dataset
input_path = "generated_dataset/counterfact_sor_edited.json"
with open(input_path, "r") as file:
    data = json.load(file)

# Function to generate a question using OpenAI API
def generate_questions(prompt_text, retries=3, delay=5):
    for attempt in range(retries):
        try:
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
        except openai.error.APIError as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise  # Reraise the exception if max retries are exceeded

# Generate the new dataset with the required format
output_dataset = []

for item in tqdm(data, desc="Processing entries"):
    case_id = item["case_id"]
    subject = item.get("subject", "")
    relation = item.get("relation", "")
    object_true = item.get("object_false", "")
    
    # Generate 4 questions using OpenAI API for each entry
    questions = [
        {"case_id": case_id, "question": generate_questions(f"Generate a random trivia question that is unrelated to the subject '{subject}', object '{object_true}', or relation '{relation}'.")},
        {"case_id": case_id, "question": generate_questions(f"Generate a random trivia question that is unrelated to the subject '{subject}', object '{object_true}', or relation '{relation}'.")},
        {"case_id": case_id, "question": generate_questions(f"Generate a random trivia question that is unrelated to the subject '{subject}', object '{object_true}', or relation '{relation}'.")},
        {"case_id": case_id, "question": generate_questions(f"Generate a random trivia question that is unrelated to the subject '{subject}', object '{object_true}', or relation '{relation}'.")}
    ]
    
    # Add the questions to the output dataset
    output_dataset.extend(questions)

# Output path
output_path = "generated_dataset/counterfact_generality_edited.json"

# Save the generated dataset to a JSON file
with open(output_path, "w") as file:
    json.dump(output_dataset, file, indent=4)

print(f"Dataset generation completed. Results saved to '{output_path}'.")

