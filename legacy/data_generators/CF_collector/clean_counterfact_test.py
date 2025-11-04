# Load the uploaded dataset
file_path = "generated_dataset/counterfact_sor_cleaned.json"

# Read the file content
with open(file_path, "r") as file:
    data_content = file.read()


import json

try:
    dataset = json.loads(data_content)
except json.JSONDecodeError as e:
    dataset = None
    error_message = str(e)

# Check if the dataset was loaded successfully
if dataset:
    # List to collect case IDs that do not meet the requirements
    invalid_cases = []

    # Loop through each case in the dataset
    for entry in dataset:
        case_id = entry.get("case_id")
        prompt_for_subject = entry.get("prompt_for_subject", [])
        prompt_for_object_false = entry.get("prompt_for_object_false", [])
        subject = entry.get("subject", "")
        object_false = entry.get("object_false", "")

        # Check the conditions
        if not all(object_false in prompt for prompt in prompt_for_subject):
            invalid_cases.append(case_id)
        if not all(subject in prompt for prompt in prompt_for_object_false):
            invalid_cases.append(case_id)

    # Output invalid cases
    print("Invalid Cases:", invalid_cases)
