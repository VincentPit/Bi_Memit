import json
import random
import requests
import openai
import time
from requests.exceptions import ChunkedEncodingError

openai.api_key = ""

def fetch_neighbors(entity, retries=3, delay=5):
    query = f"""
    SELECT ?property ?propertyLabel ?valueLabel WHERE {{
      ?entity ?p ?statement.
      ?statement ?ps ?value.
      ?entity rdfs:label "{entity}"@en.
      ?property wikibase:directClaim ?ps.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 50
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "KnowledgeEditor/1.0 (example@example.com)"}

    for attempt in range(retries):
        try:
            response = requests.get(url, params={"query": query, "format": "json"}, headers=headers, timeout=10)
            response.raise_for_status()  # Raises HTTPError for bad responses
            data = response.json()
            neighbors = [
                (item["propertyLabel"]["value"], item["valueLabel"]["value"])
                for item in data["results"]["bindings"]
            ]
            return neighbors
        except ChunkedEncodingError as e:
            print(f"ChunkedEncodingError: Attempt {attempt + 1} failed - {e}")
        except requests.exceptions.RequestException as e:
            # Handle other types of request-related exceptions
            print(f"RequestException: Attempt {attempt + 1} failed - {e}")

        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2  # Exponential backoff

    # If all retries fail
    return []



# def fetch_neighbors(entity):
#     query = f"""
#     SELECT ?property ?propertyLabel ?valueLabel WHERE {{
#       ?entity ?p ?statement.
#       ?statement ?ps ?value.
#       ?entity rdfs:label "{entity}"@en.
#       ?property wikibase:directClaim ?ps.
#       SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
#     }}
#     LIMIT 50
#     """
#     url = "https://query.wikidata.org/sparql"
#     headers = {"User-Agent": "KnowledgeEditor/1.0 (example@example.com)"}
#     response = requests.get(url, params={"query": query, "format": "json"}, headers=headers)
#     if response.status_code == 200:
#         data = response.json()
#         neighbors = [
#             (item["propertyLabel"]["value"], item["valueLabel"]["value"])
#             for item in data["results"]["bindings"]
#         ]
#         return neighbors
#     else:
#         return []


def generate_prompts(relation, obj, retries=3, delay=5):
    """
    Generate prompts for GPT-4 based on few-shot learning.
    Args:
        relation (str): The relation connecting subject and object.
        obj (str): The object entity.
    Returns:
        tuple: Generated prompts for subject, relation, and object.
    """
    few_shot_subject = f"""
Generate a question for the given inputs. 
The question should reference the relation and the object, 
the answer to the question should be a subject.
Add constraints to the question to make sure the answer to your generated question is one and only, 
without mentioning the subject itself in the question.

Input:

Relation: is the manufacturer of
Object: Colt's Manufacturing Company
Questions:
What Colt weapon is a double-action revolver that was originally released in 1986 and brought back due to popular demand?, 

Input:

Relation: {relation}
Object: {obj}
Question:
""".strip()

    for attempt in range(retries):
        try:
            subject_response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for generating questions based on inputs."},
                    {"role": "user", "content": few_shot_subject}
                ],
                max_tokens=150,
            )
            prompt_for_subject = subject_response["choices"][0]["message"]["content"].strip().split("\n")
            return prompt_for_subject
        except openai.error.APIError as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
                delay *= 2  # Exponential backoff
            else:
                raise


def generate_questions(entity, entity_type, neighbors, case_id):
    """
    Generate questions based on neighbors for the given entity.
    Args:
        entity (str): The subject or object name.
        entity_type (str): 'subject' or 'object'.
        neighbors (list): List of tuples (relation, neighbor).
        case_id (int): Case ID from the dataset.
    Returns:
        list: A list of question entries in the required format.
    """
    questions = []
    selected_neighbors = random.sample(neighbors, min(3, len(neighbors)))
    for relation, neighbor in selected_neighbors:
        prompt_for_subject = generate_prompts(relation, entity)
        questions.append({"case_id": case_id, "question": prompt_for_subject[0]})
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
        obj_true = entry["object_true"]
        obj_false = entry["object_false"]

        # Fetch neighbors for the subject and object
        subject_neighbors = fetch_neighbors(subject)
        object_true_neighbors = fetch_neighbors(obj_true)
        object_false_neighbors = fetch_neighbors(obj_false)

        # Generate questions
        subject_questions = generate_questions(subject, "subject", subject_neighbors, case_id)
        object_true_questions = generate_questions(obj_true, "object", object_true_neighbors, case_id)
        object_false_questions = generate_questions(obj_false, "object", object_false_neighbors, case_id)
        

        

        all_questions.extend(subject_questions)
        all_questions.extend(object_true_questions)
        all_questions.extend(object_false_questions)


    return all_questions

if __name__ == "__main__":
    dataset_path = "generated_dataset/counterfact_sor_cleaned.json"
    output_path = "generated_dataset/counterfact_locality_edited.json"

    questions = process_dataset(dataset_path)

    with open(output_path, "w") as output_file:
        json.dump(questions, output_file, indent=4)

    print(f"Questions generated and saved to {output_path}")



