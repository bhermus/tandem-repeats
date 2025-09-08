import torch
import pandas as pd
import os
import numpy as np
import re
import argparse

from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import transformers
from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification, TrainingArguments
from datasets import Dataset


def set_cuda():
    print("Number of GPUs: ", torch.cuda.device_count())
    print("GPU Name: ", torch.cuda.get_device_name())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)


def get_non_repeats(path_to_file):
    max_lines = 30000000
    with open(path_to_file, 'r') as file:
        lines = file.readlines(max_lines)
    data = "".join([line.strip() for line in lines if line[0] != ">" and len(line.strip()) > 25])
    txt_sequences = re.split(r'N+', data)
    print("Loaded non-repeats:", len(txt_sequences))
    return txt_sequences


def get_repeats(path_to_file):
    vcf_sequences = []
    max_lines = 30000
    with open(path_to_file, 'r') as file:
        for i, line in enumerate(file):
            if i > max_lines:
                break
            if not line.startswith('#'):
                columns = line.strip().split('\t')
                if len(columns) > 3:
                    vcf_sequences.append(columns[3])
    print("Loaded repeats:", len(vcf_sequences))
    return vcf_sequences


def make_dataset(non_repeats, repeats, max_size=None):
    if max_size is not None:
        half_size = max_size // 2
        non_repeats = non_repeats[:half_size]
        repeats = repeats[:half_size]
    data = [(seq, 0) for seq in non_repeats] + [(seq, 1) for seq in repeats]
    return pd.DataFrame(data, columns=['sequence', 'label'])


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "matthews_correlation": matthews_corrcoef(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits[0] if isinstance(logits, tuple) else logits
    return calculate_metric_with_sklearn(logits, labels)


class TandemTrainer:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.train_dataset = self.test_dataset = self.val_dataset = None

    def preprocess(self):
        def tokenize(df):
            tokens = self.tokenizer(df["sequence"], padding="max_length", truncation=True, return_tensors="pt")
            return {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "labels": torch.tensor(df["label"], dtype=torch.long),
            }

        train_df, test_df = train_test_split(self.dataset, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

        self.train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
        self.test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)
        self.val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

    def train(self, training_args):
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            args=training_args,
        )
        trainer.train()

    def predict(self, sequences):
        self.model.eval()
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predictions = torch.argmax(logits, dim=-1)
        return predictions.tolist()

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)


def main():
    parser = argparse.ArgumentParser(description="Train DNA classifier")
    parser.add_argument("--non_repeat_file", required=True)
    parser.add_argument("--repeat_file", required=True)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--max_size", type=int, default=3000)
    parser.add_argument("--model_dir", default="PoetschLab/GROVER")
    parser.add_argument("--dataset_csv", default="dna_dataset.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_cuda()

    if os.path.exists(args.dataset_csv):
        dataset = pd.read_csv(args.dataset_csv)
    else:
        non_repeats = get_non_repeats(args.non_repeat_file)
        repeats = get_repeats(args.repeat_file)
        dataset = make_dataset(non_repeats, repeats, max_size=args.max_size)
        dataset.to_csv(args.dataset_csv, index=False)

    if os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")):
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    tt = TandemTrainer(model=model, tokenizer=tokenizer, dataset=dataset)
    if not os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")):
        tt.preprocess()
        tt.train(TrainingArguments(
            seed=42,
            output_dir=args.output_dir,
            logging_strategy="epoch",
            per_device_train_batch_size=16,
            evaluation_strategy="epoch",
            learning_rate=1e-6,
            num_train_epochs=10,
        ))
        tt.save_model(args.output_dir)

    predictions = tt.predict([
        "ACTGGAGGGTCTGCAGGCAGGTACCTGGGTGCTGGAGGGTCTGCAGGCAGGTACCTGGG",
        "TTTGGGCCC",
    ])
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
