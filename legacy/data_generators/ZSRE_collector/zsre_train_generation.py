import argparse
import json
import requests
import openai

# OpenAI API Key (use environment variable or config in practice)
openai.api_key = "enter your key"

# Argument parser
parser = argparse.ArgumentParser(description="Relation extraction and prompt generation.")
parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Specify mode: train or eval")
#parser.add_argument("--output_file", type=str, default="output_results.json", help="File path to store the results")
args = parser.parse_args()


# Load dataset
data_path = f"memit_datasets/zsre_mend_{args.mode}.json"
with open(data_path, "r") as file:
    data = json.load(file)

# Initialize OpenAI API for relation extraction
multi_shot_prompt = """
Extract the relation from this question:

Example:
Question: "Who directed the movie <SUBJECT>?"
Relation: "direct"
---
Question: "Where is <SUBJECT> located?"
Relation: "is located in"
---
Question: "When was the <SUBJECT> built?"
Relation: "is built"
---
Question: "What was the capital of <SUBJECT>?"
Relation: "is the capital of"
---
Now, extract the relation for the following question. Answer only the relation in no more than a few words and use present tense.
"""

def extract_relation(question_text):
    """Extract the relation from a given question using OpenAI API."""
    prompt = multi_shot_prompt + f"Question: \"{question_text}\"\nRelation:"
    
    # Call OpenAI API to get the relation
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating questions based on inputs."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
    )
    
    # Extract the relation from the API response
    relation = response["choices"][0]["message"]["content"].strip().split("\n")
    
    return relation[0].strip()

def get_wikidata_id(relation):
    """Fetch the Wikidata ID for a given relation by progressively shortening the relation while maintaining word order."""
    endpoint = "https://www.wikidata.org/w/api.php"
    words = relation.split()
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            short_relation = " ".join(words[start:start + length])
            params = {
                "action": "wbsearchentities",
                "search": short_relation,
                "language": "en",
                "format": "json",
                "type": "property"
            }
            response = requests.get(endpoint, params=params)
            data = response.json()
            if data.get("search"):
                return data["search"][0]["id"]
    return None

def generate_prompts(subject, relation, obj):
    """Generate prompts for subject, relation, and object."""
    few_shot_subject = """
Generate questions for the given inputs. 
The questions should reference the relation and the object, 
and the answer to the question should be a subject.
Add constraints to the quesiton to make sure the answer to your generated question is one and only, 
without mentioning the subject itself in the question

Input:

Relation: is the manufacturer of
Object: Colt's Manufacturing Company
Questions:
What Colt weapon is a double-action revolver that was originally released in 1986 and brought back due to popular demand?, 
What Colt revolver, originally launched in the late 1980s, is known for its stainless steel finish and .357 Magnum chambering?
What handgun manufactured by Colt was named after a snake and is designed for precision shooting?

Input:

Relation: {relation}
Object: {obj}


1. make sure there is one and only answer to the questions
2. make sure to include exactly the object, the word {obj} literally appear in the questions

Questions: 


""".strip().format(relation=relation, obj=obj)
    
    few_shot_object = """
Generate questions for the given inputs. 
The questions should reference the relation and the subject, 
and the answer to the question should be an object.

Input:
Subject: Colt King Cobra
Relation: is the manufacturer of
Object:
Questions:
 Who is the manufacturer of Colt King Cobra?,
 Which company manufactures Colt King Cobra?,
 What company is the manufacturer of Colt King Cobra?,
 Which company manufactured Colt King Cobra?,
 Who produces Colt King Cobra?,
 What is the name of the manufacturer for Colt King Cobra?

Input:
Subject: {subject}
Relation: {relation}

1. make sure there is one and only answer to the questions
2. make sure to include exactly the object, the word {subject} literally appear in the questions

Questions: 
""".strip().format(subject=subject, relation=relation)
    
    subject_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating questions based on inputs."},
            {"role": "user", "content": few_shot_subject}
        ],
        max_tokens=150,
    )
    
    prompt_for_subject = subject_response["choices"][0]["message"]["content"].strip().split("\n")
    
    object_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating questions based on inputs."},
            {"role": "user", "content": few_shot_object}
        ],
        max_tokens=150,
    )
    
    prompt_for_object = object_response["choices"][0]["message"]["content"].strip().split("\n")
    
    prompt_for_relation = [
        f"What is the relationship between {subject} and {obj}?",
    ]
    
    return prompt_for_subject, prompt_for_relation, prompt_for_object

def get_relation(src_question):
    """Process an individual entry for relation extraction and prompt generation."""
    
    relation = extract_relation(src_question)
    relation_id = get_wikidata_id(relation)
    
    if not relation_id:
        print(f"No Wikidata ID found for relation: {relation}")
        return None, None
    
    return relation, relation_id

def process_balanced_entry(subject, relation, pred_object, relation_id):
    """Generate balanced entries for relation types."""
    prompt_for_subject, prompt_for_relation, prompt_for_object = generate_prompts(subject, relation, pred_object)
    
    return {
        "subject": subject,
        "relation": relation,
        "object_true": pred_object,
        "object_false": pred_object,
        "prompt_for_subject": prompt_for_subject,
        "prompt_for_relation": prompt_for_relation,
        "prompt_for_object": prompt_for_object,
        "relation_id": relation_id,
        "source": "zsre"
    }

with open('count_question_train.json', 'r') as file:
    fre_data = json.load(file)

# Filter questions with more than 10 appearances
filtered_data = {k: v for k, v in fre_data.items() if v > 10}

# Sort the questions by frequency (ascending)
sorted_data = sorted(filtered_data.items(), key=lambda x: x[1])

num_frequent = 40
num_infrequent = 40

# Select the frequent questions (last part of the sorted list)
frequent_questions = sorted_data[-num_frequent:] if len(sorted_data) >= num_frequent else sorted_data[-len(sorted_data):]

# Select the infrequent questions (first part of the sorted list)
infrequent_questions = sorted_data[:num_infrequent] if len(sorted_data) >= num_infrequent else sorted_data[:len(sorted_data)]

# Convert back to dictionary format
frequent_questions_dict = dict(frequent_questions)
infrequent_questions_dict = dict(infrequent_questions)

# Print the results
print("Frequent Questions (appeared more than 10 times):")
print(frequent_questions_dict)

print("\nInfrequent Questions (appeared more than 10 times):")
print(infrequent_questions_dict)

def find_data(question, data):
    count = 0
    result = []
    for entry in data:
        if count == 10:
            break
        subject = entry.get("subject", "")
        src = entry.get("src", "")
        assert subject != '' and src != ''
        
        # Replace subject with a placeholder
        src = src.replace(subject, "<SUBJECT>")
        if src == question:
            count += 1
            result.append(entry)
    return result


balanced_results = []
count_fre = 0
for question in frequent_questions_dict.keys():
    if count_fre == 20:
        break
    
    relation, relation_id = get_relation(question)
    
    if relation_id:
        relation_data = find_data(question, data)
        for idx, entry in enumerate(relation_data):
            new_entry = process_balanced_entry(entry["subject"], relation, entry["pred"], relation_id)
            balanced_results.append(new_entry)
        count_fre += 1
        
count_infre = 0
for question in infrequent_questions_dict.keys():
    if count_infre == 20:
        break
    
    relation, relation_id = get_relation(question)
    
    if relation_id:
        relation_data = find_data(question, data)
        for idx, entry in enumerate(relation_data):
            new_entry = process_balanced_entry(entry["subject"], relation, entry["pred"], relation_id)
            balanced_results.append(new_entry)
        count_infre += 1
        
        
for idx in range(len(balanced_results)):
    balanced_results[idx]["case_id"] = idx+1

# Save results to a file
with open(f"output_{args.mode}_results.json", "w") as file:
    json.dump(balanced_results, file, indent=4)

print(f"Processed {len(balanced_results)} balanced entries. Results saved to out_{args.mode}_results.json.")
