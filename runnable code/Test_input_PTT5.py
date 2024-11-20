import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set the path to your saved model and tokenizer
output_dir = './t5_humor_pt_model'

# Load the trained model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(output_dir)
model = T5ForConditionalGeneration.from_pretrained(output_dir)

# Move model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")

# Function to generate predictions for custom input
def generate_custom_prediction(input_text):
    input_text = "Humor generation: " + input_text
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    # Generate the output (humorous version)
    outputs = model.generate(input_ids, max_length=256, num_beams=2)
    predicted_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return predicted_output

# Main function for user interaction
def main():
    print("Portuguese humor generation using T5!")
    print("Transform your input text into humorous text!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter a sentence: ")

        if user_input.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

        humorous_output = generate_custom_prediction(user_input)
        print(f"Humorous version: {humorous_output}\n")

if __name__ == "__main__":
    main()
