import json
import requests

# Load the dataset
file_path = "generated_dataset/zsre_sor_cleaned.json"
with open(file_path, 'r') as f:
    dataset = json.load(f)

# Function to get the entity ID from Wikidata by searching for the label
def get_entity_id(label):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "format": "json",
        "limit": 1
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "search" in data and len(data["search"]) > 0:
            return data["search"][0]["id"]
        else:
            print(f"Entity with label '{label}' not found on Wikidata.")
    else:
        print(f"Failed to fetch data for '{label}': HTTP {response.status_code}")

    return None

# Iterate over each entry in the dataset and add the subject_id and object_id
for entry in dataset:
    # Get subject and object labels
    subject_label = entry.get("subject")
    object_label = entry.get("object_true")  
    
    # Add subject_id if it doesn't exist or is empty
    if "subject_id" not in entry or not entry["subject_id"]:
        subject_id = get_entity_id(subject_label)
        if subject_id:
            entry["subject_id"] = subject_id

    # Add object_id if it doesn't exist or is empty
    if "object_id" not in entry or not entry["object_id"]:
        object_id = get_entity_id(object_label)
        if object_id:
            entry["object_id"] = object_id

# Save the updated dataset
updated_file_path = "generated_dataset/zsre_sor_edited.json"
with open(updated_file_path, 'w') as f:
    json.dump(dataset, f, indent=4)

updated_file_path
