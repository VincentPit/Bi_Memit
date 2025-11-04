import argparse
import json
import torch
from tqdm import tqdm
from transformers import pipeline, AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up argument parsing for model selection
parser = argparse.ArgumentParser(description="Relation Extraction with Model Selection")
parser.add_argument("--model", choices=["gpt2-xl", "gpt-j", "gpt-neox", "mpt-7b", "llama-2"], default="gpt2-xl", help="Choose the model for relation extraction.")
args = parser.parse_args()

# Load dataset
with open("memit_datasets/zsre_mend_eval.json", "r") as file:
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

# Load BERT tokenizer and model for embedding generation
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

# Multi-shot prompt for relation extraction
multi_shot_prompt = """
Extract the relation from this question:

Example:
Question: "Who directed the movie Titanic?"
Relation: "directed the movie"
---
Question: "Where is the Eiffel Tower located?"
Relation: "is located in"
---
Now, extract the relation for the following question: 

Answer only the relation. Should not be more than a few words.
"""

# Function to extract relation from question
def extract_relation(question_text):
    prompt = multi_shot_prompt + f"Question: \"{question_text}\"\nRelation:"
    result = relation_extractor(prompt, max_new_tokens=100, truncation=True, num_return_sequences=1, do_sample=False)
    relation = result[0]['generated_text'].split("Relation:")[-1].strip()
    return relation.split("\n")[0].strip()

# Threshold for cosine similarity in clustering
SIMILARITY_THRESHOLD = 0.85

# Dictionary to store grouped relations with their embeddings
relation_groups = {}

# Function to embed a relation using BERT
def embed_relation(relation_text):
    inputs = tokenizer(relation_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()

# Function to find or create a relation group based on similarity
def find_or_create_relation_group(relation, embedding):
    for group_relation, group_embedding in relation_groups.items():
        similarity = cosine_similarity(embedding.unsqueeze(0).cpu(), group_embedding.unsqueeze(0).cpu())[0][0]
        if similarity >= SIMILARITY_THRESHOLD:
            return group_relation  # Use existing group if similarity threshold is met
    
    # If no similar group found, add the new relation as a group
    relation_groups[relation] = embedding
    return relation

# Dictionary to store relation counts
relation_counts = {}

# Process data in batches with progress bar
BATCH_SIZE = 16
for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Processing batches"):
    batch = data[i:i + BATCH_SIZE]
    
    for item in batch:
        # Use 'src' field if available, otherwise fall back to 'rephrase'
        question_text = item.get("src") or item.get("rephrase")
        
        # Extract relation
        if question_text:
            relation = extract_relation(question_text)
            embedding = embed_relation(relation)  # Embed relation with BERT
            standardized_relation = find_or_create_relation_group(relation, embedding)  # Find or create relation group
            
            # Save relation and count
            item["relation"] = standardized_relation
            relation_counts[standardized_relation] = relation_counts.get(standardized_relation, 0) + 1

# Save data and relation counts to files
output_path = "mend_dataset_count"
with open(f"{output_path}/zsre_mend_eval_count.json", "w") as file:
    json.dump(data, file, indent=4)
with open(f"{output_path}/relation_counts.json", "w") as file:
    json.dump(relation_counts, file, indent=4)

print("Relation extraction and counting completed. Results saved to 'mend_dataset_count/zsre_mend_eval_count.json' and 'relation_counts.json'.")
