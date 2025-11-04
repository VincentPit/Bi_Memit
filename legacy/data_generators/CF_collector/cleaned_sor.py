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

# Load the uploaded file to process
file_path = "generated_dataset/zsre_sor_cleaned.json"

# Read the content of the dataset file
with open(file_path, "r") as file:
    file_content = file.read()

# OpenAI API key 
openai.api_key = ""
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


try:
    start_idx = file_content.find("[")
    end_idx = file_content.rfind("]") + 1
    dataset_json = file_content[start_idx:end_idx]
    dataset = json.loads(dataset_json)
except Exception as e:
    print(f"Error while extracting JSON: {e}")
    dataset = None

# Placeholder for invalid cases list 
invalid_cases = [15, 20, 21, 32, 33, 39, 48, 66, 71, 72, 85, 123, 158, 160, 161, 168, 181, 184, 194, 196, 197, 198, 201, 204, 213, 231, 241, 252, 276, 291, 296, 306, 318, 375]

# Regenerate questions for invalid cases
updated_dataset = []

if dataset:
    for entry in dataset:
        if entry.get("case_id") in invalid_cases:
            # Extract the relevant details
            subject = entry["subject"]
            relation = entry["relation"]
            object_true = entry["object_true"]
            object_false = entry["object_false"]

            # Regenerate questions
            prompt_for_subject = f"Given the subject '{subject}', relation '{relation}', and an object '{object_false}', generate a natural question that asks towards the subject. The subject {subject} itself and {object_true} should not appear in the question, but {object_false} must appear in the question. The goal is to craft a question that tests knowlege of this subject in a natural way and make sure the answer to your generated question is {subject}. The question must be straightforward without multi-hops. For example, if given 'subject': 'Danielle Darrieux', 'relation': 'native language',  'object': 'English', generate questions like 'Whose native language is English?'. Also, make sure the answer to your generated question is unique, that means you have to add more constrains (never use this contriant: {object_true} and make sure {object_true} is not in your generated question) to your generated qeustion without mentioning the subject itself ({subject}) in your question so that there will be only one answer. For example, if your question asks 'what is the largest phone company?' and your expected answer is 'Huawei', then you should refine your question to 'What is the largest phone company in China?', this will let the others give the expected answer."
            prompt_for_object = f"Given the subject '{subject}',  relation '{relation}', and an object '{object_false}', generate a natural question that focuses on the object. For example, if given 'subject': 'Danielle Darrieux', 'relation': 'native language', and 'object': 'English', generate questions like 'What is Danielle Darrieux's native language?' The object {object_false} itself should not appear in the question, but {subject} must appear in the question as part of constraint or relevant information. Make sure the answer to your generated question is {object_false} and the question is straightforward without multi-hops. The goal is to craft a question that tests knowlege of this object in a natural way. Also, make sure the answer to your generated question is unique,that means you have to add more constrains to your generated qeustion without mentioning the object itself ({object_false}) in your question so that there will be only one answer.For example, if your question asks 'what is the largest phone company?' and your expected answer is 'Huawei', then you should refine your question to 'What is the largest phone company in China?', this will let the others give the expected answer."

           # Generate variations of questions
            variation_subject1 = generate_questions(f"Generate a variant question for: {prompt_for_subject}. The subject {subject} itself and {object_true} should not appear in the question, but {object_false} (not its variant, I need the exact word {object_false}) must appear in the question. Make sure the answer to your generated question is {subject}.")
            variation_subject2 = generate_questions(f"Generate a different variant of the following question: {prompt_for_subject}. The subject {subject} itself and {object_true} should not appear in the question, but {object_false} (not its variant, I need the exact word {object_false}) must appear in the question. Make sure the answer to your generated question is {subject}.")
            variation_subject3 = generate_questions(f"Generate another different variant of the following question: {prompt_for_subject}. The subject {subject} itself and {object_true} should not appear in the question, but {object_false} (not its variant, I need the exact word {object_false}) must appear in the question. Make sure the answer to your generated question is {subject}.")

            variation_object_false1 = generate_questions(f"Generate a variant question for: {prompt_for_object}. The object {object_false} itself should not appear in the question, but {subject} (not its variant, I need the exact word {subject})must appear in the question. Make sure the answer to your generated question is {object_false}.")
            variation_object_false2 = generate_questions(f"Generate a different variant of the following question: {prompt_for_object}. The object {object_false} itself should not appear in the question, but {subject} (not its variant, I need the exact word {subject}) must appear in the question. Make sure the answer to your generated question is {object_false}.")
            variation_object_false3 = generate_questions(f"Generate another different variant of the following question: {prompt_for_object}. The object {object_false} itself should not appear in the question, but {subject} (not its variant, I need the exact word {subject}) must appear in the question. Make sure the answer to your generated question is {object_false}.")
            
            entry["prompt_for_subject"] = [generate_questions(prompt_for_subject), variation_subject1, variation_subject2, variation_subject3]
            entry["prompt_for_object"] = [generate_questions(prompt_for_object), variation_object_false1, variation_object_false2, variation_object_false3]

        updated_dataset.append(entry)

    # Save the updated dataset
    output_path = "generated_dataset/zsre_sor_cleaned.json"
    with open(output_path, "w") as outfile:
        json.dump(updated_dataset, outfile, indent=4)

    print(f"Updated dataset saved to {output_path}")
else:
    print("Dataset extraction failed.")
