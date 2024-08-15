"""
Train the quality classifier.
This is the third and final stage of the pipeline.
"""
import sqlite3

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import Dataset
from scipy.stats import spearmanr
from sklearn.metrics import (cohen_kappa_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (XLMRobertaModel, XLMRobertaTokenizerFast,
                          get_linear_schedule_with_warmup)

from model import XLMRobertaForRegression


# Load data from the SQLite db
def load_data(db_path: str):
    texts, scores = [], []
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT text, quality_score FROM evaluated_texts")
    for row in c.fetchall():
        texts.append(row[0])
        scores.append(row[1])
    conn.close()
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

def evaluate_model(model, eval_dataloader):
    """
    Calculate evaluation metrics for the model.
    Metrics:
    - MAE provides an intuitive measure of average error.
    - Weighted Kappa accounts for the ordinal nature of your data.
    - Spearman's correlation shows how well your model ranks the texts.
    - R² gives you an idea of how much variance your model explains.
    - Macro-MAE helps you understand if your model performs consistently across all score levels.
    If MAE is low but Kappa is also low, it might indicate that your model is making 
      small but systematically incorrect predictions.
    If Spearman's correlation is high but MAE is not great, it suggests 
      that your model ranks texts well but might need calibration for absolute predictions.
    Comparing Macro-MAE to MAE can reveal if your model performs poorly on certain score levels.
    """
    model.eval()
    eval_loss = 0
    eval_preds = []
    eval_labels = []
    
    eval_pbar = tqdm(eval_dataloader, desc="Evaluating", leave=False, position=2)
    
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
    
    # Round predictions to nearest integer for classification metrics
    eval_preds_rounded = np.round(eval_preds).astype(int)
    eval_labels_rounded = np.round(eval_labels).astype(int)
    
    # Regression metrics
    mae = mean_absolute_error(eval_labels, eval_preds)
    mse = mean_squared_error(eval_labels, eval_preds)
    r2 = r2_score(eval_labels, eval_preds)
    
    # Classification metrics
    weights = 'quadratic'  # Penalizes larger disagreements more
    kappa = cohen_kappa_score(eval_labels_rounded, eval_preds_rounded, weights=weights)
    
    # Spearman's Rank Correlation (works with continuous values)
    spearman_corr, _ = spearmanr(eval_labels, eval_preds)
    
    # Macro-averaged MAE
    unique_labels = np.unique(eval_labels_rounded)
    macro_mae = np.mean([
        mean_absolute_error(eval_labels[eval_labels_rounded == label], eval_preds[eval_labels_rounded == label])
        for label in unique_labels
    ])
    
    return {
        "Loss": eval_loss,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "Weighted Kappa": kappa,
        "Spearman Correlation": spearman_corr,
        "Macro-MAE": macro_mae
    }


def main():
    wandb.init(project="tekstkvalitet", name="xlm-roberta-base")

    accelerator = Accelerator(
        # dynamo_backend="inductor" # doesn't work on old GPUs
        )

    # Set seed for reproducibility
    set_seed(42)

    # Load pre-trained model and tokenizer
    model = XLMRobertaForRegression.from_pretrained('xlm-roberta-base', num_unfrozen_layers=1)
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')

    # Load all data
    texts, scores = load_data('evaluated_texts.db')

    # Normalize scores to [0, 1] range
    scores = np.array(scores) / 5.0

    # Split data into train and test sets
    train_texts, test_texts, train_scores, test_scores = train_test_split(texts, scores, test_size=0.02, random_state=42)

    # Prepare train and test datasets
    train_dataset = prepare_sliding_window_dataset(tokenizer, train_texts, train_scores)
    test_dataset = prepare_sliding_window_dataset(tokenizer, test_texts, test_scores)

    # Define training parameters
    num_epochs = 2
    train_batch_size = 16
    eval_batch_size = 64
    learning_rate = 1e-4
    weight_decay = 0.01
    num_warmup_steps = 50
    lambda_reg = 0.01 # regularization strength towards original params
    eval_steps = 500

    # Log hyperparameters to wandb
    wandb.config.update({
        "num_epochs": num_epochs,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_warmup_steps": num_warmup_steps,
        "lambda_reg": lambda_reg,
        "eval_steps": eval_steps
    })

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, collate_fn=collate_fn)
    eval_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=collate_fn)

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

    # Training loop with hierarchical progress bars
    global_step = 0
    best_mae = float('inf')
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
    for epoch in epoch_pbar:
        model.train()
        step_pbar = tqdm(range(num_update_steps_per_epoch), desc=f"Epoch {epoch+1}", leave=False, position=1)
        for _ in step_pbar:
            batch = next(iter(train_dataloader))
            outputs = model(**batch)
            loss = outputs[0]
            reg_loss = model.get_param_regularization_loss(lambda_reg)
            total_loss = loss + reg_loss
            accelerator.backward(total_loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            step_pbar.set_postfix({
                    "Train loss": f"{loss.item():.4f}",
                    "Reg loss": f"{reg_loss.item():.4f}",
                })
            
            # Log training metrics to wandb
            wandb.log({
                "train_loss": loss.item(),
                "reg_loss": reg_loss.item(),
                "total_loss": total_loss.item(),
                "learning_rate": lr_scheduler.get_last_lr()[0]
            }, step=global_step)
            
            global_step += 1

            if global_step % eval_steps == 0:
                metrics = evaluate_model(model, eval_dataloader)

                # Print detailed metrics
                print(f"\nEpoch {epoch+1}, Step {global_step} Evaluation Metrics:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")

                # Log evaluation metrics to wandb
                wandb.log(metrics, step=global_step)

                # Save the best model
                if metrics['MAE'] < best_mae:
                    best_mae = metrics['MAE']
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(unwrapped_model.state_dict(), "./model-best.safetensors", safe_serialization=True)
                    print(f"New best model saved with MAE: {best_mae:.4f}")
                    wandb.log({"best_mae": best_mae}, step=global_step)

    # Final evaluation
    print("\nPerforming final evaluation...")
    final_metrics = evaluate_model(model, eval_dataloader)
    print("\nFinal Evaluation Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Log final metrics to wandb
    wandb.log(final_metrics, step=global_step)

    # Save the final model in Hugging Face format
    print("\nSaving the final model...")
    output_dir = "./model-final"
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save the model configuration with custom code information
    config = unwrapped_model.config
    config.num_labels = 1
    config.problem_type = "regression"
    config.auto_map = {"AutoModel": "model.XLMRobertaForRegression"}
    config.custom_objects = {"create_model": "model.create_model"}
    config.save_pretrained(output_dir)

    # Save the custom model code
    import shutil
    shutil.copy("model.py", f"{output_dir}/model.py")

if __name__ == "__main__":
    main()
