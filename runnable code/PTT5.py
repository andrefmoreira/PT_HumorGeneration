import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_from_disk, DatasetDict
import os
import gc

# Clear GPU cache if needed
gc.collect()
torch.cuda.empty_cache()

# Paths
dataset_dir = './processed_datasets'
output_dir = './t5_humor_pt_model'

# Load preprocessed datasets
train_dataset = load_from_disk(os.path.join(dataset_dir, "train"))
validation_dataset = load_from_disk(os.path.join(dataset_dir, "validation"))
test_dataset = load_from_disk(os.path.join(dataset_dir, "test"))

# Combine datasets into a DatasetDict
tokenized_dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")
model = T5ForConditionalGeneration.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")

if torch.cuda.is_available():
    model.to("cuda")

# Preprocessing function
def preprocess_data(examples):
    inputs = ["Humor generation: " + original for original in examples['original']]
    targets = examples['transformed']

    # Tokenize inputs and outputs
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=256, truncation=True, padding='max_length')

    # Replace padding token IDs in labels with -100
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq] 
        for label_seq in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Tokenize datasets
tokenized_train_dataset = tokenized_dataset["train"].map(preprocess_data, batched=True, load_from_cache_file=False)
tokenized_validation_dataset = tokenized_dataset["validation"].map(preprocess_data, batched=True, load_from_cache_file=False)

# Set training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Adjust batch size based on your system's capacity
    per_device_eval_batch_size=4,
    num_train_epochs=6,
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    save_strategy="epoch",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
)

# Train the model
trainer.train()

# Save the final model locally
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Evaluate the model on the validation dataset
eval_results = trainer.evaluate()

# Print the evaluation results
print(f"Evaluation results: {eval_results}")

# Clear cache after training
torch.cuda.empty_cache()
