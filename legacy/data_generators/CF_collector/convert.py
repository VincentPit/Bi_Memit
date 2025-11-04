import json

# Load the generated dataset
input_path = "generated_dataset/counterfact_generality.json"
with open(input_path, "r") as file:
    data = json.load(file)

# Extract only the questions from the dataset, maintaining the part after '\n\n'
extracted_questions = []
for item in data:
    question = item["question"].split("\n\n", 1)[-1].strip()
    extracted_questions.append({"case_id": item["case_id"], "question": question})

# Output path for the extracted questions
output_path = "generated_dataset/extracted_questions.json"

# Save the extracted questions to a new JSON file
with open(output_path, "w") as file:
    json.dump(extracted_questions, file, indent=4)

print(f"Extraction completed. Results saved to '{output_path}'.")
