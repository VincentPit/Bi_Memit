import json
import random
import requests
import openai

openai.api_key = "enter your key"

def generate_prompts(obj):
   
    few_shot_subject = f"""
Generate 3 random simple single-hop questions that has nothting to with the subject. 
Here is an example

Input:
Subject: Colt's Manufacturing Company

Questions:
Who is the president of United States?
What sport was Michael Jordan Playing?
When did China come into a nation?
What sports did Roger Federer play?
(the subject or its neighbors, attributes should not even appear in the question)
Input:

Subject: {obj}
Question:
""".strip()

    subject_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating questions based on inputs."},
            {"role": "user", "content": few_shot_subject}
        ],
        max_tokens=150,
    )
    prompt_for_subject = subject_response["choices"][0]["message"]["content"].strip().split("\n")

    return prompt_for_subject


def generate_questions(entity, case_id):
    questions = []

    prompt_for_subject = generate_prompts(entity)
    for question in prompt_for_subject:
        questions.append({"case_id": case_id, "question": question})
    return questions


def process_dataset(dataset_path):
    """
    Process the input dataset and generate questions for testing in knowledge editing.
    Args:
        dataset_path (str): Path to the input JSON file.
    Returns:
        list: Generated question entries.
    """
    with open(dataset_path, "r") as file:
        data = json.load(file)

    all_questions = []

    for entry in data:
        case_id = entry["case_id"]
        subject = entry["subject"]
        obj = entry["object_true"]

        # Generate questions
        subject_questions = generate_questions(subject, case_id)
        object_questions = generate_questions(obj, case_id)

        all_questions.extend(subject_questions)
        all_questions.extend(object_questions)

    return all_questions

if __name__ == "__main__":
    dataset_path = "output_train_results.json" 
    output_path = "generality_questions.json"

    questions = process_dataset(dataset_path)

    with open(output_path, "w") as output_file:
        json.dump(questions, output_file, indent=4)

    print(f"Questions generated and saved to {output_path}")
