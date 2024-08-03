"""
Train the quality classifier.
This is the third and final stage of the pipeline.
"""
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=1)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Prepare your dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = examples["rating"]
    return tokenized_inputs

# Assume 'texts' is your list of Danish documents and 'ratings' is your list of corresponding ratings (1-5)
dataset = Dataset.from_dict({"text": texts, "rating": ratings})
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Convert ratings to float
tokenized_dataset = tokenized_dataset.cast_column("labels", torch.float)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Define custom data collator
def collate_fn(examples):
    return tokenizer.pad(examples, padding=True, return_tensors="pt")

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(range(len(tokenized_dataset) // 10)),  # Use 10% of data for evaluation
    data_collator=collate_fn,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./multilingual-quality-classifier")
tokenizer.save_pretrained("./multilingual-quality-classifier")
