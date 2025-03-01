import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json

MODEL_NAME = "google/flan-t5-large"  # Using Flan-T5
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Define available function names
FUNCTION_NAMES = [
    "find_current_location",
    "calculate_fastest_route",
    "navigate_to_nearest_charging_station"
]

# Function to classify prompt into one of the function names
def classify_function(prompt):
    system_prompt = f"You are an AI assistant that maps user prompts to function names. Choose from: {', '.join(FUNCTION_NAMES)}."
    input_text = f"{system_prompt}\nUser: {prompt}\nAssistant:"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)

    # Decode model output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract function name from response (assuming it's the first valid match)
    for function in FUNCTION_NAMES:
        if function in response:
            return function
    
    return "Unknown function"

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

# Run classification
for prompt, true_function in test_prompts:
    predicted_function = classify_function(prompt)
    print(f"Prompt: {prompt}\nTrue Function: {true_function}\nPredicted Function: {predicted_function}\n")
    if predicted_function == true_function:
        correct_count += 1

total_prompts = len(test_prompts)
accuracy_percentage = (correct_count / total_prompts) * 100

print(f"Accuracy: {correct_count}/{total_prompts} ({accuracy_percentage:.2f}%)")