import requests
import urllib3

def get_wikidata_id(subject_name):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": subject_name
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data['search']:
            # Return the first search result's ID
            return data['search'][0]['id']
        else:
            print("No Wikidata ID found for the subject:", subject_name)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Wikidata ID: {e}")
        return None




def get_entity_data(entity_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def get_relation_label(relation_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{relation_id}.json"
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        data = response.json()
        return data['entities'][relation_id]['labels']['en']['value']
    except requests.exceptions.RequestException as e:
        print(f"Error fetching relation label: {e}")
        return None

def extract_entity_relations(entity_id):
    data = get_entity_data(entity_id)
    if data is None:
        return []

    entity_data = data['entities'][entity_id]
    entity_relations = []

    subject = {
        "name": entity_data['labels']['en']['value'],
        "description": entity_data['descriptions'].get('en', {}).get('value', ''),
        "wiki_id": entity_id
    }

    # Extract relationships from claims
    for prop_id, claims in entity_data['claims'].items():
        relation_label = get_relation_label(prop_id)  # Fetch human-readable relation label

        for claim in claims:
            mainsnak = claim.get('mainsnak', {})
            if mainsnak.get('datatype') == 'wikibase-item':
                related_entity_id = mainsnak['datavalue']['value']['id']
                related_data = get_entity_data(related_entity_id)
                if related_data:
                    related_entity = related_data['entities'][related_entity_id]
                    object_entity = {
                        "name": related_entity['labels']['en']['value'],
                        "description": related_entity['descriptions'].get('en', {}).get('value', ''),
                        "wiki_id": related_entity_id
                    }

                    # Add to relations list
                    entity_relations.append({
                        "subject": subject,
                        "relation": relation_label,  # Use the human-readable relation label
                        "object": object_entity
                    })
    return entity_relations




def show_all_relation(entity_id):
    relations_with_context = extract_entity_relations(entity_id)

    # Display each subject-relation-object with rich context and IDs
    for relation in relations_with_context:
        print("Subject:", relation["subject"]["name"])
        print("Description:", relation["subject"]["description"])
        print("Wiki ID:", relation["subject"]["wiki_id"])
        print("Relation:", relation["relation"])
        print("Object:", relation["object"]["name"])
        print("Object Description:", relation["object"]["description"])
        print("Object Wiki ID:", relation["object"]["wiki_id"])
        print("-" * 40)

def get_entities_by_relation(property_id, target_entity_id):
    url = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/json"
    }
    
    # Define the SPARQL query
    query = f"""
    SELECT ?subject ?subjectLabel WHERE {{
      ?subject wdt:{property_id} wd:{target_entity_id}.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """

    try:
        response = requests.get(url, headers=headers, params={'query': query})
        response.raise_for_status()
        data = response.json()
        
        # Extract the list of entities with the specified relation
        entities = []
        for item in data['results']['bindings']:
            entities.append({
                "subject": item['subject']['value'],  # URL of the entity
                "subjectLabel": item['subjectLabel']['value']  # Label of the entity
            })
        
        return entities
    except requests.exceptions.RequestException as e:
        print(f"Error fetching entities by relation: {e}")
        return None




if __name__ = "__main___":
    # Example usage with a known Wikidata ID for "Python (programming language)"
    entity_id = "Q28865"  # Wikidata ID for Python (programming language)
    
    # Example usage
    subject_name = "Python (programming language)"
    wikidata_id = get_wikidata_id(subject_name)
    print(f"Wikidata ID for '{subject_name}': {wikidata_id}")
    
    
    property_id = "P31"  # Instance of
    target_entity_id = "Q146"  # Target entity (e.g., House cat)
    entities = get_entities_by_relation(property_id, target_entity_id)

    # Display the entities with their labels
    for entity in entities:
        print(f"Entity: {entity['subject']}, Label: {entity['subjectLabel']}")