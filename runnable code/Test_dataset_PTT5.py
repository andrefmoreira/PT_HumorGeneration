import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_from_disk
import os

# Set the paths
output_dir = './t5_humor_pt_model'
dataset_dir = './processed_datasets'

# Load the test dataset from the preprocessed and saved files
test_dataset = load_from_disk(os.path.join(dataset_dir, "test"))

# Load the trained model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(output_dir)
model = T5ForConditionalGeneration.from_pretrained(output_dir)

# Move model to GPU if available
if torch.cuda.is_available():
    model.to('cuda')

# Define a function to generate predictions using the trained model
def generate_predictions(test_data):
    predictions = []
    for example in test_data:
        input_text = "Humor generation: " + example['original']
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids

        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        # Generate the output (humorous version)
        outputs = model.generate(input_ids, max_length=256, num_beams=2)
        predicted_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append((input_text, predicted_output, example['transformed']))

    return predictions

# Generate predictions on the test dataset
predictions = generate_predictions(test_dataset)

# Evaluate predictions
correct_predictions = 0
total_predictions = len(predictions)
different = []
different_input = []
different_output = []

# Compare the generated predictions to the expected outputs
for input_text, predicted_output, expected_output in predictions:
    print(f"Input: {input_text}")
    print(f"Prediction: {predicted_output}")
    print(f"Expected: {expected_output}")
    print("-" * 50)
    
    cleaned_predicted_output = predicted_output.replace("Humor generation: ", "").strip()
    cleaned_input_text = input_text.replace("Humor generation: ", "").strip()

    if cleaned_predicted_output != cleaned_input_text:
        different.append(cleaned_predicted_output)
        different_input.append(cleaned_input_text)
        different_output.append(expected_output)

    if predicted_output.strip() == expected_output.strip():
        correct_predictions += 1

# Calculate and print the accuracy
accuracy = correct_predictions / total_predictions * 10
for i in range(len(different)):
    print(f"Different from input" )
    print(different[i])    
    print(different_input[i])
    print(different_output[i])
    
print(f"Test accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions} correct)")
