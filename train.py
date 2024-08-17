import os
import json
import pandas as pd
import numpy as np
import optuna
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AdamW, get_linear_schedule_with_warmup, DataCollatorWithPadding
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import datetime
import logging
import sys

# Function to calculate and print the number of trainable parameters in the model
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return (
        f"trainable model parameters: {trainable_model_params}\n"
        f"all model parameters: {all_model_params}\n"
        f"percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
    )

# Load and prepare data
json_dir = '/home/aviv/Desktop/deep_learning/dp/Article-Bias-Prediction/data/jsons'
data_list = []

# Store original stdout and stderr for later use
original_stdout = sys.stdout
original_stderr = sys.stderr

# Load JSON files from the specified directory and prepare the data
for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        with open(os.path.join(json_dir, filename), 'r') as file:
            data = json.load(file)
            # Rename the 'bias' key to 'labels'
            if 'bias' in data:
                data['labels'] = data.pop('bias')
            data_list.append(data)

# Convert the list of data to a DataFrame
df = pd.DataFrame(data_list)

# Define column names for text and labels
text_column = 'content_original'  # or use 'title'
label_column = 'bias_text'

# Map textual bias labels to numerical values
label_map = {'left': 0, 'center': 1, 'right': 2}
df[label_column] = df[label_column].map(label_map)

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['content_original'], df['labels'], test_size=0.15, stratify=df['labels'], random_state=42
)

# Combine texts and labels into DataFrames for training and testing
train_df = pd.DataFrame({'content_original': train_texts, 'labels': train_labels})
test_df = pd.DataFrame({text_column: test_texts, 'labels': test_labels})

# Reset indices of DataFrames
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Further split the training set into training and validation sets
train_df, val_df = train_test_split(train_df, test_size=3/17, random_state=42)
val_df.reset_index(drop=True, inplace=True)
train_df.reset_index(drop=True, inplace=True)

# Display the shapes of the resulting DataFrames
print(f"Training set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Initialize the model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

# Print the number of trainable parameters in the base model
print(print_number_of_trainable_model_parameters(model))

# Define the LoRA Configuration
lora_config = LoraConfig(
    r=128,  # Rank number
    lora_alpha=8,  # Scaling factor
    lora_dropout=0,  # Dropout probability for LoRA
    target_modules=["q_lin", "k_lin", "v_lin"],  # Apply LoRA to specific layers
    bias='none',
    task_type=TaskType.SEQ_CLS  # Sequence classification task
)

# Apply LoRA configuration to the model
peft_model = get_peft_model(model, lora_config)

# Print the number of trainable parameters after applying LoRA
print(print_number_of_trainable_model_parameters(peft_model))

# Function to tokenize input data
def tokenize_func(data):
    return tokenizer(
        data[text_column],
        max_length=512,  # Adjustable max length
        padding='max_length',
        return_attention_mask=True,
        truncation=True
    )

# Convert DataFrames to Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_func, batched=True, remove_columns=[text_column])
val_dataset = val_dataset.map(tokenize_func, batched=True, remove_columns=[text_column])
test_dataset = test_dataset.map(tokenize_func, batched=True, remove_columns=[text_column])

# Define evaluation metrics
def metrics(eval_prediction):
    logits, labels = eval_prediction
    pred = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='macro', zero_division=1)
    accuracy = accuracy_score(labels, pred)
    report = classification_report(labels, pred)
    print("Classification Report:\n", report)
    return {"f1_acc": f1, "accuracy": accuracy}

# Hyperparameter configurations
train_batch_size = 32
weight_decays = [5e-4] #[5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 10e-4, 11e-4, 12e-4, 13e-4, 14e-4, 15e-4]
learning_rates = [5e-4]
lora_r_vals = [64]
lora_alpha = 128
drop_outs = [0] #[0, 2e-5, 4e-5, 8e-5, 16e-5, 32e-5]

# Loop through combinations of hyperparameters for model training
for weight_decay in weight_decays:
    for drop_out in drop_outs:
        for lora_r in lora_r_vals:
            for lr in learning_rates:

                # Re-initialize model for each hyperparameter combination
                model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=3)
                eval_batch_size = train_batch_size
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
                n_epochs = 5
                total_steps = n_epochs * math.ceil(len(train_dataset) / train_batch_size)

                # Print the number of trainable parameters in the base model
                print(print_number_of_trainable_model_parameters(model))

                # Update LoRA configuration with current hyperparameters
                lora_config = LoraConfig(
                    r=lora_r,  # Rank number
                    lora_alpha=lora_alpha,  # Scaling factor
                    lora_dropout=drop_out,  # Dropout probability for LoRA
                    target_modules=["q_lin", "k_lin", "v_lin"],
                    bias='none',
                    task_type=TaskType.SEQ_CLS  # Sequence classification task
                )

                # Apply LoRA to the model
                peft_model = get_peft_model(model, lora_config)

                # Print the number of trainable parameters after applying LoRA
                print(print_number_of_trainable_model_parameters(peft_model))

                # Define training arguments
                peft_training_args = TrainingArguments(
                    output_dir='./result-distilbert-lora',
                    logging_dir='./logs-distilbert-lora',
                    learning_rate=lr,
                    per_device_train_batch_size=train_batch_size,  # Adjust based on GPU memory
                    per_device_eval_batch_size=eval_batch_size,  # Adjust based on GPU memory
                    num_train_epochs=n_epochs,
                    logging_steps=total_steps / n_epochs,
                    evaluation_strategy='steps',
                    eval_steps=total_steps / n_epochs,
                    weight_decay=weight_decay,
                    seed=42,
                    fp16=True,  # Use FP16 if running on GPU
                    report_to='none',
                    disable_tqdm=True  # Disable progress bars
                )

                # Define optimizer
                optimizer = AdamW(peft_model.parameters(), lr=lr, no_deprecation_warning=True)

                # Define learning rate scheduler
                lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=total_steps
                )

                # Define data collator
                collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

                # Initialize the Trainer
                peft_trainer = Trainer(
                    model=peft_model,
                    args=peft_training_args,
                    train_dataset=train_dataset,  # Training data
                    eval_dataset=val_dataset,  # Validation data
                    tokenizer=tokenizer,
                    compute_metrics=metrics,
                    optimizers=(optimizer, lr_scheduler),
                    data_collator=collator
                )

                # Generate a unique filename for logging
                filename = (
                    f"output_Lora-r{lora_config.r}alpha{lora_config.lora_alpha}"
                    f"dropout{lora_config.lora_dropout}Adam-lr{peft_trainer.args.learning_rate}"
                    f"epochs{peft_trainer.args.num_train_epochs}wd{peft_trainer.args.weight_decay}"
                    f"trainbatchsize{train_batch_size}.log3"
                )

                # Add a timestamp to the filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"{timestamp}_{filename}"

                # Set up logging to a file
                logging.basicConfig(
                    filename=filename,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                )

                # Log dataset shapes
                logging.info(f"Training set shape: {train_df.shape}")
                logging.info(f"Validation set shape: {val_df.shape}")
                logging.info(f"Test set shape: {test_df.shape}")

                # Open the log file and redirect stdout and stderr to it
                log_file = open(filename, 'w')
                sys.stdout = log_file
                sys.stderr = log_file

                # Print and log information about the training process
                print(f"Training set shape: {train_df.shape}")
                print(f"Validation set shape: {val_df.shape}")
                print(f"Test set shape: {test_df.shape}")
                print(f"Total Steps: {total_steps}")

                # Define path to save the fine-tuned model
                peft_model_path = "/home/aviv/Desktop/deep_learning/dp/models"

                # Train the model
                peft_trainer.train()

                # Evaluate and save the model
                peft_trainer.evaluate(test_dataset)
                peft_trainer.save_model(peft_model_path)

                # Close the log file and reset stdout and stderr to original
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                log_file.close()

                # Notify that the training for this configuration is done
                print(f"{filename} - DONE")