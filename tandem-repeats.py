import torch
import pandas as pd
import matplotlib.pyplot as plt

import wget
import os
import numpy as np
import re

from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


import transformers
from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification, TrainingArguments, AutoModelForMaskedLM
from datasets import Dataset

pd.set_option("display.max_rows", 100)  # Allow showing at least 100 rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_colwidth", None)  # No truncation for column values


def set_cuda():
    print("Number of GPUs: ", torch.cuda.device_count())
    print("GPU Name: ", torch.cuda.get_device_name())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)


def get_non_repeats(path_to_file):
    # Read the .txt file
    max_lines = 30000000

    with open(path_to_file, 'r') as file:
        lines = file.readlines(max_lines)

    data = "".join([line.strip() for line in lines if line[0] != ">" and len(line.strip()) > 25])

    txt_sequences = re.split(r'N+', data)  # Split at consecutive 'N's

    # Print sample output
    print(txt_sequences[:10])  # View first 10 sequences
    print("Total sequences:", len(txt_sequences))

    return txt_sequences


def get_repeats(path_to_file):
    # Read the .vcf file
    vcf_sequences = []
    i = 0
    max_lines = 30000
    with open(path_to_file, 'r') as file:
        for line in file:
            i += 1
            if i > max_lines:
                break
            if not line.startswith('#'):  # Skip metadata/header lines
                columns = line.strip().split('\t')
                if len(columns) > 3:
                    vcf_sequences.append(columns[3])  # Reference sequence column

    # Print sample output
    print(vcf_sequences[:10])  # View first 10 sequences
    print(min(len(seq) for seq in vcf_sequences))  # Minimum length of sequences
    print("Total sequences: ", len(vcf_sequences))

    return vcf_sequences


def make_dataset(non_repeats, repeats, max_size=None):
    if max_size is not None:
        half_size = max_size // 2
        non_repeats = non_repeats[:half_size]
        repeats = repeats[:half_size]

    # Combine into a dataset
    data = [(seq, 0) for seq in non_repeats] + [(seq, 1) for seq in repeats]
    dataset = pd.DataFrame(data, columns=['sequence', 'label'])

    dataset.to_csv('dna_dataset_full.csv', index=False)
    return dataset


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(logits, axis=-1)
    labels = labels.flatten() if len(labels.shape) > 1 else labels  # Ensure labels are 1D

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": matthews_corrcoef(
            labels, predictions
        ),
        "precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


class TandemTrainer:
    def __init__(
            self,
            model,
            tokenizer,
            dataset,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

        self.train_dataset = None  # Run preprocess() to set
        self.test_dataset = None  # Run preprocess() to set
        self.val_dataset = None  # Run preprocess() to set

    def preprocess(self):
        def preprocess_data(df):
            tokenized = self.tokenizer(df["sequence"], padding="max_length", truncation=True, return_tensors="pt")
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": torch.tensor(df["label"], dtype=torch.long),
            }

        # Split into train (80%), test (20%)
        train_df, test_df = train_test_split(self.dataset, test_size=0.2, random_state=42)

        # Further split train into train (80%) and validation (20%) of original training data
        train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

        # Print dataset sizes
        print(f"Train: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")

        self.train_dataset = Dataset.from_pandas(train_df)
        self.train_dataset = self.train_dataset.map(preprocess_data, batched=True, batch_size=100, keep_in_memory=False)

        self.test_dataset = Dataset.from_pandas(test_df)
        self.test_dataset = self.test_dataset.map(preprocess_data, batched=True)

        self.val_dataset = Dataset.from_pandas(val_df)
        self.val_dataset = self.val_dataset.map(preprocess_data, batched=True)

    def train(self,
              training_args=TrainingArguments(
                                seed=42,
                                output_dir="./results",
                                logging_strategy="epoch",
                                per_device_train_batch_size=16,
                                evaluation_strategy="epoch",
                                learning_rate=0.000001,
                                num_train_epochs=10,
                            )
              ):
        trainer = transformers.Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            args=training_args,
        )
        trainer.train()

    def predict(self, sequences):
        self.model.eval()  # Set model to evaluation mode
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        # Detect model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        return predictions.tolist()

    def save_model(self, path="./results"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_trained_model(self, path="./results"):
        """
        Loads a previously trained model and tokenizer from the given directory.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)


def main():
    set_cuda()

    tokenizer = transformers.AutoTokenizer.from_pretrained("PoetschLab/GROVER")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("PoetschLab/GROVER")

    try:
        dataset = pd.read_csv("dna_dataset.csv")
    except FileNotFoundError:
        non_repeats = get_non_repeats("output.txt")
        repeats = get_repeats("VNTRseek_NA19240_ERX283215.vcf")
        dataset = make_dataset(non_repeats, repeats, max_size=3000)

    tt = TandemTrainer(model=model, tokenizer=tokenizer, dataset=dataset)
    tt.preprocess()
    tt.train()


if __name__ == "__main__":
    main()
