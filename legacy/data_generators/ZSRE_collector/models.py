import json
import torch
import argparse
from SPARQLWrapper import SPARQLWrapper, JSON
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
import random

# Argument parser for model selection
parser = argparse.ArgumentParser(description="Choose model to use for response generation.")
parser.add_argument("--model", choices=["gpt2", "llama", "gptj", "gptneo"], required=True, help="Specify which model to use: gpt2, llama, gptj, or gptneo")
args = parser.parse_args()

# Load models based on user input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.model == "gpt2":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device)
elif args.model == "llama":
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")  # replace with actual model path
    model = LlamaForCausalLM.from_pretrained("huggyllama/llama-7b").to(device)
elif args.model == "gptj":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
elif args.model == "gptneo":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to(device)

# Function to fetch random context from Wikidata
def fetch_random_wikidata_context():
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = """
    SELECT ?itemLabel ?description WHERE {
      ?item wdt:P31 wd:Q5;  # Q5 for humans (you can adjust to other types if needed)
            rdfs:label ?itemLabel;
            schema:description ?description.
      FILTER (lang(?itemLabel) = "en" && lang(?description) = "en")
    }
    LIMIT 50
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    random_entry = random.choice(results['results']['bindings'])
    label = random_entry['itemLabel']['value']
    description = random_entry['description']['value']
    
    return f"{label}: {description}"

# Function to generate a response using a local model
def generate_local_response(prompt, question, model, tokenizer, max_length=100):
    input_text = f"{prompt} {question}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Annotate the dataset
def annotate_dataset(dataset):
    annotated_data = []
    for entry in dataset:
        prompt = entry['prompt']
        question = entry['question']
        
        # Fetch random context from Wikidata
        wikidata_context = fetch_random_wikidata_context()
        
        # Add Wikidata context to the prompt
        prompt_with_context = f"{wikidata_context}\n\n{prompt}"
        
        # Generate response based on selected model
        response = generate_local_response(prompt_with_context, question, model, tokenizer)
        
        # Add annotation to the dataset entry
        annotated_entry = {
            "prompt": prompt,
            "question": question,
            "context": wikidata_context,
            f"{args.model}_response": response
        }
        annotated_data.append(annotated_entry)
    
    return annotated_data

# Example dataset with prompts and questions
dataset = [
    {"prompt": "Describe the function of photosynthesis.", "question": "What is the role of chlorophyll?"},
    {"prompt": "Explain Newton's laws of motion.", "question": "How does inertia relate to the laws?"}
]

# Annotate the dataset
annotated_data = annotate_dataset(dataset)

# Convert to JSON format and save to a file
json_output = json.dumps(annotated_data, indent=4)
with open("annotated_dataset.json", "w") as f:
    f.write(json_output)

print("Dataset annotated and saved to annotated_dataset.json")
