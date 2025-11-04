import argparse
import json
import torch
from tqdm import tqdm
from transformers import pipeline, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import re
from sentence_transformers import SentenceTransformer
import gensim
import numpy as np

# Argument parser for model, embedding choice, and mode (train or eval)
parser = argparse.ArgumentParser(description="Relation Extraction with Model Selection and Embedding Choice")
parser.add_argument("--model", choices=["gpt2-xl", "gpt-j", "gpt-neox", "mpt-7b", "llama-2"], default="gpt2-xl", help="Choose the model for relation extraction.")
parser.add_argument("--embedding", choices=["MiniLM", "GloVe", "Word2Vec", "BERT", "None"], default="None", help="Choose the embedding method for relations.")
parser.add_argument("--mode", choices=["train", "eval"], default="eval", help="Specify mode: train or eval")
args = parser.parse_args()

# Load dataset based on mode
data_path = f"memit_datasets/zsre_mend_{args.mode}.json"
with open(data_path, "r") as file:
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
    #print("Extracted relation:", relation)
    return relation.split("\n")[0].strip()

# Initialize embedding model based on user's choice
if args.embedding == "MiniLM":
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
elif args.embedding == "GloVe":
    glove_embeddings = {}
    with open("glove.6B.50d.txt", "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            glove_embeddings[word] = vector
elif args.embedding == "Word2Vec":
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("word2vec-google-news-300.bin", binary=True)
elif args.embedding == "BERT":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

def get_glove_embedding(text):
    words = text.split()
    vectors = [glove_embeddings[word] for word in words if word in glove_embeddings]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(50)

def get_word2vec_embedding(text):
    words = text.split()
    vectors = [word2vec_model[word] for word in words if word in word2vec_model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(300)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings

# Function to embed relation based on user's choice
def embed_relation(relation_text):
    if args.embedding == "MiniLM":
        return torch.tensor(embedding_model.encode(relation_text)).to(device)
    elif args.embedding == "GloVe":
        return torch.tensor(get_glove_embedding(relation_text)).to(device)
    elif args.embedding == "Word2Vec":
        return torch.tensor(get_word2vec_embedding(relation_text)).to(device)
    elif args.embedding == "BERT":
        return get_bert_embedding(relation_text).to(device)

SIMILARITY_THRESHOLD = 0.85 if args.embedding != "None" else None
relation_groups = {}

# Modified function to find or create relation groups
def find_or_create_relation_group(relation, embedding=None):
    if args.embedding == "None":  # No embedding, group by exact match
        
        print("using Nome embed")
        if relation in relation_groups:
            return relation
        else:
            relation_groups[relation] = None
            return relation
    else:  # Use embeddings for grouping
        for group_relation, group_embedding in relation_groups.items():
            similarity = cosine_similarity(embedding.unsqueeze(0).cpu(), group_embedding.unsqueeze(0).cpu())[0][0]
            if similarity >= SIMILARITY_THRESHOLD:
                return group_relation
        relation_groups[relation] = embedding
        return relation

relation_counts = {}
BATCH_SIZE = 32
for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Processing batches"):
    batch = data[i:i + BATCH_SIZE]
    
    for item in batch:
        question_text = item.get("src") or item.get("rephrase")
        if question_text:
            relation = extract_relation(question_text)
            
            # Embedding only if not "None"
            embedding = embed_relation(relation) if args.embedding != "None" else None
            
            standardized_relation = find_or_create_relation_group(relation, embedding)
            item["relation"] = standardized_relation
            relation_counts[standardized_relation] = relation_counts.get(standardized_relation, 0) + 1

output_path = f"/scratch/jl13122/KeDataCollector/mend_dataset_count"
with open(f"{output_path}/zsre_mend_{args.mode}_count.json", "w") as file:
    json.dump(data, file, indent=4)
with open(f"{output_path}/{args.mode}_relation_counts.json", "w") as file:
    json.dump(relation_counts, file, indent=4)

print(f"Relation extraction and counting completed. Results saved to '{output_path}/zsre_mend_{args.mode}_count.json' and '{output_path}/{args.mode}_relation_counts.json'.")
