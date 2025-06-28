from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load your dataset
dataset = load_dataset('csv', data_files={'train': 'data/train.csv'}, delimiter=',')

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Tokenize the data
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Define training arguments with logging
training_args = TrainingArguments(
    output_dir='./model/saved_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",  # change to 'steps' if you add eval data
    report_to="none",  # disable WANDB or other integrations
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./law-model")
tokenizer.save_pretrained("./law-model")

print("✅ Training complete. Model saved to './law-model'")
