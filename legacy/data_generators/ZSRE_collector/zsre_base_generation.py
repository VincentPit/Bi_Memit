import torch
from transformers import pipeline
import json

def generate_questions_few_shot(subject, relation, obj, model_name="gpt2-xl"):
    """
    Generate questions based on subject, relation, and object using few-shot learning and a specified language model.
    
    Args:
        subject (str): The subject of the triple.
        relation (str): The relation between the subject and object.
        obj (str): The object of the triple.
        model_name (str): The name of the language model to use for generation.

    Returns:
        dict: A dictionary containing the generated questions and their expected answers.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the specified language model
    generator = pipeline("text-generation", model=model_name, device=0 if device == "cuda" else -1)

    # Few-shot examples for subject-relation to object question generation
    subject_few_shot_examples = """
Generate a clear and concise question where the answer is the object. Do not include examples or explanations in your response.
Just provide the question.

Example 1:
Subject: "Isaac Newton"
Relation: "discovered"
Object: "gravity"
Question: "What did Isaac Newton discover?"

Example 2:
Subject: "Marie Curie"
Relation: "won"
Object: "two Nobel Prizes"
Question: "What did Marie Curie win?"

Example 3:
Subject: "Alexander Fleming"
Relation: "invented"
Object: "penicillin"
Question: "What did Alexander Fleming invent?"

Now, generate a question for the following:
Subject: "{subject}"
Relation: "{relation}"
Object: "{obj}"
Question:
"""

    # Few-shot examples for object-relation to subject question generation
    object_few_shot_examples = """
Generate a clear and concise question where the answer is the subject. Do not include examples or explanations in your response.
Just provide the question.

Example 1:
Subject: "Isaac Newton"
Relation: "discovered"
Object: "gravity"
Question: "Who discovered gravity?"

Example 2:
Subject: "Marie Curie"
Relation: "won"
Object: "two Nobel Prizes"
Question: "Who won two Nobel Prizes?"

Example 3:
Subject: "Alexander Fleming"
Relation: "invented"
Object: "penicillin"
Question: "Who invented penicillin?"

Now, generate a question for the following:
Object: "{obj}"
Relation: "{relation}"
Subject: "{subject}"
Question:
"""

    # Format prompts with the given subject, relation, and object
    subject_prompt = subject_few_shot_examples.format(subject=subject, relation=relation, obj=obj)
    object_prompt = object_few_shot_examples.format(subject=subject, relation=relation, obj=obj)

    # Generate questions using the model
    subject_question_response = generator(subject_prompt, max_new_tokens=100, num_return_sequences=1, do_sample=False)
    object_question_response = generator(object_prompt, max_new_tokens=100, num_return_sequences=1, do_sample=False)

    # Extract and clean up the generated questions
    subject_question_raw = subject_question_response[0]['generated_text']
    object_question_raw = object_question_response[0]['generated_text']

    subject_question = subject_question_raw.split("Question:")[-1].strip()
    object_question = object_question_raw.split("Question:")[-1].strip()

    # Create a dictionary to store the results
    results = {
        "subject_relation_question": subject_question,
        "object_relation_question": object_question,
        "expected_answers": {
            "subject_relation_question": obj,
            "object_relation_question": subject
        },
        "raw_answers": {
            "subject_relation_question_raw": subject_question_raw,
            "object_relation_question_raw": object_question_raw
        }
    }

    return results


# Example Usage
if __name__ == "__main__":
    # Define the subject, relation, and object
    subject = "Albert Einstein"
    relation = "developed"
    obj = "the theory of relativity"

    # Test the function with different models
    models = ["gpt2-xl"]  # Add other models like "meta-llama/Llama-2-7b-hf" if needed

    for model in models:
        results = generate_questions_few_shot(subject, relation, obj, model_name=model)

        # Save the results to a file
        filename = f"results_{model.replace('/', '_')}.json"
        with open(filename, "w") as file:
            json.dump(results, file, indent=4)

        print(f"Results saved to {filename}")
        print(f"Raw Subject Relation Question: {results['raw_answers']['subject_relation_question_raw']}")
        print(f"Raw Object Relation Question: {results['raw_answers']['object_relation_question_raw']}")
