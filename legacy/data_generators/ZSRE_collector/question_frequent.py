import json

# Load data from the JSON file
with open('count_question_train.json', 'r') as file:
    data = json.load(file)

# Filter questions with more than 10 appearances
filtered_data = {k: v for k, v in data.items() if v > 10}

# Sort the questions by frequency (ascending)
sorted_data = sorted(filtered_data.items(), key=lambda x: x[1])

# Handle cases where there are fewer than 20 frequent or infrequent questions
num_frequent = 20
num_infrequent = 20

# Select the frequent questions (last part of the sorted list)
frequent_questions = sorted_data[-num_frequent:] if len(sorted_data) >= num_frequent else sorted_data[-len(sorted_data):]

# Select the infrequent questions (first part of the sorted list)
infrequent_questions = sorted_data[:num_infrequent] if len(sorted_data) >= num_infrequent else sorted_data[:len(sorted_data)]

# Convert back to dictionary format
frequent_questions_dict = dict(frequent_questions)
infrequent_questions_dict = dict(infrequent_questions)

# Print the results
print("Frequent Questions (appeared more than 10 times):")
print(frequent_questions_dict)

print("\nInfrequent Questions (appeared more than 10 times):")
print(infrequent_questions_dict)
