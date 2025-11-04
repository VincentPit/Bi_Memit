import requests
import json
import torch
#from transformers import pipeline
import openai


openai.api_key = "enter your key"

# Relation extractor initialization
#relation_extractor = pipeline("text-generation", model="gpt2-xl", device=0 if torch.cuda.is_available() else -1)

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

def extract_relation(question_text):
    """Extract the relation from a given question using OpenAI API."""
    prompt = multi_shot_prompt + f"Question: \"{question_text}\"\nRelation:"
    
    
    """subject_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating questions based on inputs."},
            {"role": "user", "content": few_shot_subject}
        ],
        max_tokens=150,
    )"""
    # OpenAI API to get the relation
    response = openai.ChatCompletion.create(
        model="gpt-4",  #  "gpt-3.5-turbo" or "gpt-4" 
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating questions based on inputs."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
    )
    
    print("Raw Result:", response)
    
    # Extract the relation from the API response
    #subject_response["choices"][0]["message"]["content"].strip().split("\n")
    relation = response["choices"][0]["message"]["content"].strip().split("\n")
    
    return relation[0].strip()

def get_wikidata_id(relation):
    """Fetch the Wikidata ID for a given relation by progressively shortening the relation while maintaining word order."""
    # API endpoint
    endpoint = "https://www.wikidata.org/w/api.php"
    
    # Split the relation into individual words or parts
    words = relation.split()
    
    # Generate all possible consecutive word combinations (maintaining order)
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            # Get the consecutive substring by taking 'length' words starting from 'start'
            short_relation = " ".join(words[start:start + length])
            
            # Set parameters for the Wikidata API search
            params = {
                "action": "wbsearchentities",
                "search": short_relation,
                "language": "en",
                "format": "json",
                "type": "property"
            }
            
            # Make the request
            response = requests.get(endpoint, params=params)
            data = response.json()
            
            # Check if a match is found and return the ID of the first match
            if data.get("search"):
                return data["search"][0]["id"]  # Return the first matched ID
    
    # If no valid ID is found, return None
    return None

def clean_prompt_format(prompt_list):
    """Helper function to clean and format prompt list."""
    cleaned_prompts = [p.strip().strip('",') for p in prompt_list]
    return cleaned_prompts

def generate_prompts(subject, relation,  obj): #reversed_relation,
    """Generate prompts for subject, relation, and object."""
    few_shot_subject = """
Generate questions for the given inputs. 
The questions should reference the relation and the object, 
and the answer to the question should be a subject.
Add constraints to the quesiton to make sure the answer to your generated question is one and only, 
without mentioning the subject itself ({subject}) in the question

Input:

Subject:
Relation: is the manufacturer of
Object: Colt's Manufacturing Company
Questions:
What Colt weapon is a double-action revolver that was originally released in 1986 and brought back due to popular demand?, 
What Colt revolver, originally launched in the late 1980s, is known for its stainless steel finish and .357 Magnum chambering?
What handgun manufactured by Colt was named after a snake and is designed for precision shooting?

Input:

Subject: 
Relation: {relation}
Object: {obj}
Questions: (make sure there is one and only answer to the question)
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
Object:
Questions:  (make sure there is one and only answer to the question)
""".strip().format(subject=subject, relation=relation)
    
    subject_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating questions based on inputs."},
            {"role": "user", "content": few_shot_subject}
        ],
        max_tokens=150,
    )
    """subject_response = openai.Completion.create(
        model="gpt-4",
        prompt=few_shot_subject,
        max_tokens=150
    )"""
    prompt_for_subject = subject_response["choices"][0]["message"]["content"].strip().split("\n")
    
    object_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating questions based on inputs."},
            {"role": "user", "content": few_shot_object}
        ],
        max_tokens=150,
    )
    """object_response = openai.Completion.create(
        model="gpt-4",
        prompt=few_shot_object,
        max_tokens=150
    )"""
    prompt_for_object = object_response["choices"][0]["message"]["content"].strip().split("\n")
    
    prompt_for_relation = [
        f"What is the relationship between {subject} and {obj}?",
    ]
    
    return prompt_for_subject, prompt_for_relation, prompt_for_object



def process_entry(entry):
    """Process an individual entry for relation extraction and prompt generation."""
    subject = entry.get("subject")
    src_question = entry.get("src")
    pred_object = entry.get("pred")
    
    # Step 1: Extract the relation
    relation = extract_relation(src_question)
    
    # Step 2: Find the relation's Wikidata ID
    relation_id = get_wikidata_id(relation)
    
    if not relation_id:
        print(f"No Wikidata ID found for relation: {relation}")
        return
    
    print(f"Extracted Relation: {relation} (Wikidata ID: {relation_id})")
    
    # Step 3: Generate prompts
    prompt_for_subject, prompt_for_relation, prompt_for_object = generate_prompts(subject, relation, pred_object)
    
    # Clean up the prompts
    cleaned_subject = clean_prompt_format(prompt_for_subject)
    cleaned_relation = clean_prompt_format(prompt_for_relation)
    cleaned_object = clean_prompt_format(prompt_for_object)
    
    # Return the cleaned results
    return {
        "subject": subject,
        
        "relation": relation,
        
        "object_true": pred_object,
        "object_false": pred_object,
        
        "prompt_for_subject": cleaned_subject,
        
        "prompt_for_relation": cleaned_relation,
        
        "prompt_for_object": cleaned_object
        
        "relation_id": relation_id,
        
        "source": "zsre"
    }
    
if __name__ == "__main__":
    #Testing relation id retrival
    print("Testing relation id retrival")
    relation = "made of ABC"
    wikidata_id = get_wikidata_id(relation)
    
    if wikidata_id:
        print(f"Wikidata ID for '{relation}': {wikidata_id}")
    else:
        print(f"No Wikidata ID found for relation: '{relation}'")

    
    
    #Testing prompt generation 
    print("Testing prompt generation")
    
    subject = "Colt King Cobra"
    relation = "manufacturer"
    object = "Colt's Manufacturing Company"

    prompts = generate_prompts(subject, relation, object)
    print("Prompts for Subject:")
    print(prompts[0])
    print("\nPrompts for Relation:")
    print(prompts[1])
    print("\nPrompts for Object:")
    print(prompts[2])
    
    
    # Test general input data
    print("Testing general input data")
    entry = {
        "subject": "Colt King Cobra",
        "src": "The manufacturer of Colt King Cobra was who?",
        "pred": "Colt's Manufacturing Company",
        "rephrase": "Which company did Colt King Cobra produce?",
        "alt": "Colt's Manufacturing Corporation",
        "answers": ["Colt's Manufacturing Company"],
        "loc": "nq question: ek veer ki ardaas veera meaning in english",
        "loc_ans": "A Brother's Prayer... Veera",
        "cond": "Colt's Manufacturing Company >> Colt's Manufacturing Corporation || The manufacturer of Colt King Cobra was who?"
    }

    # Process the entry
    result = process_entry(entry)

    # Output the result
    if result:
        print(json.dumps(result, indent=4))

