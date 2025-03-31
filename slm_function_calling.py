import torch
import time
import json
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Define available function names
FUNCTION_NAMES = [
    "find_current_location",
    "calculate_fastest_route",
    "navigate_to_nearest_charging_station"
]

# Function to classify prompt into one of the function names & measure time per token
def classify_function(prompt):
    system_prompt = f"You are an AI assistant that maps user prompts to function names. Choose from: {', '.join(FUNCTION_NAMES)}."
    input_text = f"{system_prompt}\nUser: {prompt}\nAssistant:"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)

    end_time = time.time()

    total_time = end_time - start_time

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predicted_function = "Unknown function"
    for function in FUNCTION_NAMES:
        if function in response:
            predicted_function = function
            break

    num_tokens_generated = outputs.shape[1]

    time_per_token = total_time / num_tokens_generated if num_tokens_generated > 0 else 0

    return predicted_function, time_per_token

# Load test prompts from JSON file
with open('/Users/darshkodwani/Documents/Darsh/Microsoft/Code/slm_drone_classifier/slm_drone_prompts.json', 'r') as file:
    test_prompts_data = json.load(file)

# Extract prompts for each function
test_prompts = []
for function in FUNCTION_NAMES:
    if function in test_prompts_data:
        for prompt in test_prompts_data[function]:
            test_prompts.append((prompt, function))

correct_count = 0
time_per_token_list = []
total_prompts = len(test_prompts)

# Run classification & track time per token
for prompt, true_function in test_prompts:
    predicted_function, time_per_token = classify_function(prompt)
    time_per_token_list.append(time_per_token)

    # Strict exact match
    is_exact_match = predicted_function.strip().lower() == true_function.strip().lower()

    print(f"Prompt: {prompt}")
    print(f"True Function: {true_function}")
    print(f"Predicted Function: {predicted_function}")
    print(f"Exact Match: {'✅' if is_exact_match else '❌'}")
    print(f"Time per Token: {time_per_token:.6f} seconds\n")

    if predicted_function == true_function:
        correct_count += 1

# Calculate overall accuracy
accuracy_percentage = (correct_count / total_prompts) * 100
average_time_per_token = np.mean(time_per_token_list) if total_prompts > 0 else 0
std_time_per_token = np.std(time_per_token_list) if total_prompts > 0 else 0

# Print final results
print(f"Accuracy: {correct_count}/{total_prompts} ({accuracy_percentage:.2f}%)")
print(f"Average Time per Token: {average_time_per_token:.6f} seconds")
print(f"Standard Deviation of Time per Token: {std_time_per_token:.6f} seconds")
