import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

# ---- Load Dataset ----
file_path = "slm_parameter_extraction.json"  # Update if file path is different
with open(file_path, "r") as f:
    dataset = json.load(f)

# ---- Select SLM (Gemma-2B or Flan-T5) ----
MODEL_NAME = "google/flan-t5-xl"  # Change to "google/flan-t5-xl" for Flan-T5

# Load the appropriate model type
if "gemma" in MODEL_NAME:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---- Instructive System Prompt ----
SYSTEM_PROMPT = """
You are an AI assistant that extracts parameters from function calls. 
Your task is to carefully read a user’s request and extract the key parameter that corresponds to a function.

In this case, the function is `calculate_fastest_route(destination)`, which requires a `destination` parameter. 
The destination is usually a **specific location**, such as a **warehouse, facility, checkpoint, factory, or logistics center**, and may include a **city or region**.

I will show you examples of the type of parameters I expect you to extract. Your response should **only contain the extracted parameter** with no additional text.

### Examples:

User: What’s the quickest way to get to the distribution center in Seattle?
Assistant: distribution center in Seattle

User: Find the fastest path to Warehouse 42 in San Francisco, avoiding bad weather.
Assistant: Warehouse 42 in San Francisco

User: How do I reach the downtown logistics hub as quickly as possible?
Assistant: downtown logistics hub

User: Get me the shortest route to the supply depot in Denver.
Assistant: supply depot in Denver

User: Calculate the fastest route to the main storage facility in Houston.
Assistant: main storage facility in Houston

Now, extract the correct destination for the following request:
User: {prompt}
Assistant:
"""

# ---- Function to Extract Parameters from SLM ----
def extract_parameter(prompt):
    input_text = SYSTEM_PROMPT.format(prompt=prompt)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# ---- Evaluate Model Performance (Exact Match + Fuzzy Match) ----
exact_match_count = 0
fuzzy_match_count = 0
total_examples = len(dataset)

for example in dataset:
    prompt = example["prompt"]
    expected_param = example["extracted_parameter"]
    
    predicted_param = extract_parameter(prompt)

    # Strict exact match
    is_exact_match = predicted_param.strip().lower() == expected_param.strip().lower()

    # Fuzzy match (checks if either string is a substring of the other)
    is_fuzzy_match = predicted_param.lower() in expected_param.lower() or expected_param.lower() in predicted_param.lower()

    print(f"Prompt: {prompt}")
    print(f"Expected: {expected_param}")
    print(f"Predicted: {predicted_param}")
    print(f"Exact Match: {'✅' if is_exact_match else '❌'}")
    print(f"Fuzzy Match: {'✅' if is_fuzzy_match else '❌'}\n")

    if is_exact_match:
        exact_match_count += 1
    if is_fuzzy_match:
        fuzzy_match_count += 1

# ---- Print Accuracy ----
exact_match_accuracy = (exact_match_count / total_examples) * 100
fuzzy_match_accuracy = (fuzzy_match_count / total_examples) * 100

print(f"SLM Parameter Extraction Accuracy (Exact Match): {exact_match_accuracy:.2f}%")
print(f"SLM Parameter Extraction Accuracy (Fuzzy Match): {fuzzy_match_accuracy:.2f}%")