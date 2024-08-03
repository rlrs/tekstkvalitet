"""
Train the quality classifier.
This is the third and final stage of the pipeline.
"""
import torch
import numpy as np
import json
import gzip
import glob
import copy
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
from datasets import Dataset, Features, Value, Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Custom model for regression with partially frozen RoBERTa weights
class XLMRobertaForRegression(torch.nn.Module):
    def __init__(self, pretrained_model_name='xlm-roberta-base', num_unfrozen_layers=3):
        super().__init__()
        self.roberta = XLMRobertaModel.from_pretrained(pretrained_model_name)
        self.regression_head = torch.nn.Linear(self.roberta.config.hidden_size, 1)
        
        # Freeze all layers except the top num_unfrozen_layers
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        for param in self.roberta.encoder.layer[-num_unfrozen_layers:].parameters():
            param.requires_grad = True

        # Store the original parameters for regularization
        self.original_params = {}
        for name, param in self.roberta.named_parameters():
            if param.requires_grad:
                self.original_params[name] = param.data.clone()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Get the [CLS] token output
        logits = self.regression_head(cls_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return (loss, logits) if loss is not None else logits

    def get_param_regularization_loss(self, lambda_reg):
        reg_loss = 0.0
        for name, param in self.roberta.named_parameters():
            if param.requires_grad:
                reg_loss += torch.sum((param - self.original_params[name]) ** 2)
        return lambda_reg * reg_loss

# Load data from .jsonl.gz files
def load_data(file_pattern):
    texts, scores = [], []
    for filename in glob.glob(file_pattern):
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                texts.append(data['text'])
                scores.append(int(data['quality_score']))
    return texts, scores

def prepare_sliding_window_dataset(tokenizer, texts, scores, max_length=512, stride=256):
    all_input_ids, all_attention_mask, all_labels = [], [], []

    for text, score in zip(texts, scores):
        tokenized = tokenizer.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        for i in range(0, len(input_ids), stride):
            chunk_input_ids = input_ids[i:i+max_length-2]  # -2 to make room for [CLS] and [SEP]
            chunk_attention_mask = attention_mask[i:i+max_length-2]

            # Add [CLS] and [SEP] tokens
            chunk_input_ids = [tokenizer.cls_token_id] + chunk_input_ids + [tokenizer.sep_token_id]
            chunk_attention_mask = [1] + chunk_attention_mask + [1]

            # Pad if necessary
            padding_length = max_length - len(chunk_input_ids)
            if padding_length > 0:
                chunk_input_ids = chunk_input_ids + [tokenizer.pad_token_id] * padding_length
                chunk_attention_mask = chunk_attention_mask + [0] * padding_length

            all_input_ids.append(chunk_input_ids)
            all_attention_mask.append(chunk_attention_mask)
            all_labels.append(score)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels
    })

# Custom collate function to convert to tensors
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.float)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict_full_text(tokenizer, texts, max_length=512, stride=256):
    all_predictions = []
    for text in texts:
        tokenized = tokenizer.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        chunk_predictions = []

        for i in range(0, len(input_ids), stride):
            chunk_input_ids = input_ids[i:i+max_length-2]  # -2 to make room for [CLS] and [SEP]
            chunk_attention_mask = attention_mask[i:i+max_length-2]

            # Add [CLS] and [SEP] tokens
            chunk_input_ids = [tokenizer.cls_token_id] + chunk_input_ids + [tokenizer.sep_token_id]
            chunk_attention_mask = [1] + chunk_attention_mask + [1]

            # Pad if necessary
            padding_length = max_length - len(chunk_input_ids)
            if padding_length > 0:
                chunk_input_ids = chunk_input_ids + [tokenizer.pad_token_id] * padding_length
                chunk_attention_mask = chunk_attention_mask + [0] * padding_length

            inputs = {
                'input_ids': torch.tensor([chunk_input_ids]),
                'attention_mask': torch.tensor([chunk_attention_mask])
            }
            with torch.no_grad():
                outputs = model(**inputs)
            chunk_predictions.append(outputs.squeeze().item())

        # Average predictions for all chunks and denormalize
        all_predictions.append(np.mean(chunk_predictions) * 5)

    return all_predictions

def main():
    # Initialize accelerator
    accelerator = Accelerator()

    # Set seed for reproducibility
    set_seed(42)

    # Load pre-trained model and tokenizer
    model = XLMRobertaForRegression()
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')

    # Load all data
    texts, scores = load_data('../quality_filtering/output/*_evaluated.jsonl.gz')

    # Normalize scores to [0, 1] range
    scores = np.array(scores) / 5.0

    # Split data into train and test sets
    train_texts, test_texts, train_scores, test_scores = train_test_split(texts, scores, test_size=0.2, random_state=42)

    # Prepare train and test datasets
    train_dataset = prepare_sliding_window_dataset(tokenizer, train_texts, train_scores)
    test_dataset = prepare_sliding_window_dataset(tokenizer, test_texts, test_scores)

    # Define training parameters
    num_epochs = 20
    batch_size = 16
    learning_rate = 3e-5
    weight_decay = 0.01
    num_warmup_steps = 20
    lambda_reg = 0.1 # regularization strength towards original params

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    eval_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Define optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.roberta.named_parameters() if p.requires_grad], "lr": learning_rate},
            {"params": model.regression_head.parameters(), "lr": learning_rate * 10}
        ],
        weight_decay=weight_decay
    )

    # Define learning rate scheduler
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Prepare everything with our `accelerator`
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Move original parameters to the correct device after prepare
    for name, param in model.roberta.named_parameters():
        if param.requires_grad:
            model.original_params[name] = model.original_params[name].to(param.device)

    # Training loop
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
    for epoch in epoch_pbar:
        model.train()
        batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training", leave=False, position=1)
        for batch in batch_pbar:
            outputs = model(**batch)
            loss = outputs[0]
            reg_loss = model.get_param_regularization_loss(lambda_reg)
            total_loss = loss + reg_loss
            accelerator.backward(total_loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            batch_pbar.set_postfix({"loss": f"{loss.item():.4f}", "reg_loss": f"{reg_loss.item():.4f}"})

        # Evaluation
        model.eval()
        eval_loss = 0
        eval_preds = []
        eval_labels = []
        eval_pbar = tqdm(eval_dataloader, desc=f"Epoch {epoch+1} - Evaluation", leave=False, position=1)
        for batch in eval_pbar:
            with torch.no_grad():
                outputs = model(**batch)
            loss, logits = outputs[:2]
            eval_loss += loss.item()
            eval_preds.extend(logits.squeeze().tolist())
            eval_labels.extend(batch["labels"].squeeze().tolist())

        eval_loss /= len(eval_dataloader)
        eval_preds = np.array(eval_preds) * 5  # Denormalize
        eval_labels = np.array(eval_labels) * 5  # Denormalize
        eval_mse = mean_squared_error(eval_labels, eval_preds)
        eval_mae = mean_absolute_error(eval_labels, eval_preds)
        eval_r2 = r2_score(eval_labels, eval_preds)

        accelerator.print(f"Epoch {epoch+1}: "
              f"Loss: {eval_loss:.4f}, "
              f"MSE: {eval_mse:.4f}, "
              f"MAE: {eval_mae:.4f}, "
              f"R2: {eval_r2:.4f}")

    # Save the model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), "./xlm-roberta-quality-regressor.pt")
    if accelerator.is_main_process:
        tokenizer.save_pretrained("./xlm-roberta-quality-regressor")

if __name__ == "__main__":
    main()
