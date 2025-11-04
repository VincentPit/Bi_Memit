import random
import wikipedia
from collections import defaultdict
import openai
import random
openai.api_key = "your-api-key-here"



# Sample Data for Wiki Entities and their Relations (for simplicity, using basic Wikipedia API)
entities = ["Albert Einstein", "Marie Curie", "Isaac Newton", "Ada Lovelace", "Charles Darwin"]

# Sample relations we care about (for simplicity, focusing on language, birth place, etc.)
relations = [
    ("mother tongue", "P103"),  # Mother tongue relation in WikiData (example)
    ("born in", "P19")  # Place of birth relation in WikiData (example)
]

# Function to fetch information from Wikipedia (or could be replaced with WikiData query)
def fetch_entity_info(entity_name):
    try:
        page = wikipedia.page(entity_name)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        return None
    except wikipedia.exceptions.HTTPTimeoutError as e:
        return None

# Generate prompts based on different types
def generate_prompts(entity_name, relation, relation_id):
    # Define a prompt for GPT to generate different types of prompts based on the entity and relation
    prompt = f"""
    Given the entity "{entity_name}" and the relation "{relation}", generate the following types of prompts:
    1. Paraphrase prompts: Create several paraphrases of the relation for the given entity.
    2. Neighborhood prompts: Generate prompts that refer to similar entities with the same relation.
    3. Attribute prompts: Create prompts that highlight the attributes related to the relation and entity.
    4. Generation prompts: Generate open-ended prompts based on the relation and entity.

    Ensure that the prompts are diverse and sound natural, and be as creative as possible. Here are a few examples:

    "requested_rewrite": {
      "prompt": "{}, which is located in",
      "relation_id": "P17",
      "target_new": {
        "str": "Sweden",
        "id": "Q34"
      },
      "target_true": {
        "str": "Spain",
        "id": "Q29"
      },
      "subject": "Autonomous University of Madrid"
    },
    "paraphrase_prompts": [
      "and Sallie Beavers Riley. Autonomous University of Madrid is located in",
      "Houston, Tex: Anson Jones Press. Autonomous University of Madrid, located in"
    ],
    "neighborhood_prompts": [
      "Biure is located in",
      "Ripoll\u00e8s, located in",
      "Ebro, in",
      "Biure, which is located in",
      "Donostia-San Sebasti\u00e1n, in",
      "Donostia-San Sebasti\u00e1n is located in",
      "Pamplona is located in",
      "Lugo, which is located in",
      "M\u00e1laga is located in",
      "Biure, in"
    ],
    "attribute_prompts": [
      "SKF is located in",
      "K\u00f6ping Municipality, in",
      "Upplands V\u00e4sby, in",
      "Motala, in",
      "Trollh\u00e4ttan, in",
      "Upplands V\u00e4sby is located in the country of",
      "Kungs\u00f6r Municipality, located in",
      "IKEA, located in",
      "T\u00e4by, located in",
      "IKEA, which is located in"
    ],
    "generation_prompts": [
      "One can get to Autonomous University of Madrid by navigating",
      "Autonomous University of Madrid's surroundings include",
      "Autonomous University of Madrid's surroundings include",
      "One can get to Autonomous University of Madrid by navigating",
      "Autonomous University of Madrid's surroundings include",
      "One can get to Autonomous University of Madrid by navigating",
      "The best restaurants around Autonomous University of Madrid include",
      "The best restaurants around Autonomous University of Madrid include",
      "Autonomous University of Madrid's surroundings include",
      "The best restaurants around Autonomous University of Madrid include"
    ]
  },
    """

    # Call OpenAI's GPT model to generate the prompts based on the input entity and relation
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can change to GPT-4 or other engines if needed
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7
    )

    # Parse the response from the model
    generated_text = response.choices[0].text.strip()

    # Extract the different types of prompts from the generated text
    try:
        # Simple split of the text based on the format
        paraphrase_prompts = [line.strip() for line in generated_text.split('Neighborhood prompts:')[0].splitlines() if line]
        neighborhood_prompts = [line.strip() for line in generated_text.split('Attribute prompts:')[0].splitlines() if line]
        attribute_prompts = [line.strip() for line in generated_text.split('Generation prompts:')[0].splitlines() if line]
        generation_prompts = [line.strip() for line in generated_text.split('Paraphrase prompts:')[0].splitlines() if line]

        return paraphrase_prompts, neighborhood_prompts, attribute_prompts, generation_prompts
    except Exception as e:
        print(f"Error while parsing response: {e}")
        return [], [], [], []

# Function to create the dataset structure
def create_data_structure():
    dataset = []
    case_id = 1
    
    for entity in entities:
        # Simulate relation fetching from a data source (for this example, using dummy values)
        for relation, relation_id in relations:
            # Fetch entity info (could be more detailed, like birth date, language, etc.)
            entity_info = fetch_entity_info(entity)
            
            if entity_info:
                # Generate the required prompts
                paraphrase_prompts, neighborhood_prompts, attribute_prompts, generation_prompts = generate_prompts(entity, relation, relation_id)
                
                # Construct the data entry
                data_entry = {
                    "case_id": case_id,
                    "pararel_idx": random.randint(1000, 5000),  # Simulate the related index
                    "requested_rewrite": {
                        "prompt": f"The {relation} of {{}} is",
                        "relation_id": relation_id,
                        "target_new": {
                            "str": "English",  # Example language, can be dynamically generated
                            "id": "Q1860"  # WikiData ID for English
                        },
                        "target_true": {
                            "str": "Dutch",  # Example true language
                            "id": "Q7411"  # WikiData ID for Dutch
                        },
                        "subject": entity
                    },
                    "paraphrase_prompts": paraphrase_prompts,
                    "neighborhood_prompts": neighborhood_prompts,
                    "attribute_prompts": attribute_prompts,
                    "generation_prompts": random.sample(generation_prompts, 10)  # Random sample of 10
                }
                
                dataset.append(data_entry)
                case_id += 1

    return dataset

# Function to display a sample of the dataset
def display_sample(dataset, num_samples=3):
    for i in range(num_samples):
        print(f"Sample {i + 1}:")
        print(dataset[i])
        print("\n")

# Create and display the dataset
dataset = create_data_structure()
display_sample(dataset)

# Save dataset to a file (optional)
import json
with open("generated_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)
