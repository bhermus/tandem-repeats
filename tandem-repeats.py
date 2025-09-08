# Import standard and scientific libraries
import torch
import pandas as pd
import matplotlib.pyplot as plt
import wget
import os
import numpy as np
import re

# Import sklearn metrics and utility functions
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Import Hugging Face Transformers and Datasets library components
import transformers
from transformers import (
    AutoTokenizer, Trainer, AutoModelForSequenceClassification,
    TrainingArguments, AutoModelForMaskedLM
)
from datasets import Dataset

# Pandas display settings for debugging or exploration
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


# Utility function to check and report CUDA availability
def set_cuda():
    print("Number of GPUs: ", torch.cuda.device_count())
    print("GPU Name: ", torch.cuda.get_device_name())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)


# Load DNA sequences from a .txt file, skipping headers (lines starting with '>') and short lines
# Returns non-repeating sequences (non-VNTR)
# TODO This function can and should be repurposed for general loading of .txt files for training, not just non-repeats
def get_non_repeats(path_to_file):
    max_lines = 30000000
    with open(path_to_file, 'r') as file:
        lines = file.readlines(max_lines)

    # Filter and concatenate all usable lines
    data = "".join([line.strip() for line in lines if line[0] != ">" and len(line.strip()) > 25])

    # Split DNA sequence on runs of 'N' (these contained repeats and have been masked)
    txt_sequences = re.split(r'N+', data)

    print(txt_sequences[:10])  # Preview first 10 sequences
    print("Total sequences:", len(txt_sequences))

    return txt_sequences


# Load sequences from a VCF file
# Returns repeating sequences
# TODO This function can and should be repurposed for general loading of .vdf files for training, not just repeats.
# TODO Can potentially be combined with above function with read-in logic changing based on file type
def get_repeats(path_to_file):
    vcf_sequences = []
    i = 0
    max_lines = 30000  # Limit dataset size; set as needed

    with open(path_to_file, 'r') as file:
        for line in file:
            i += 1
            if i > max_lines:
                break
            if not line.startswith('#'):  # Ignore VCF metadata lines
                columns = line.strip().split('\t')
                if len(columns) > 3:
                    vcf_sequences.append(columns[3])  # Reference column (usually the 4th column)

    # Print sample output
    print(vcf_sequences[:10])  # View first 10 sequences
    print(min(len(seq) for seq in vcf_sequences))  # Minimum length of sequences
    print("Total sequences: ", len(vcf_sequences))

    return vcf_sequences


# Combine repeat and non-repeat sequences into a labeled dataset
def make_dataset(non_repeats, repeats, max_size=None):
    if max_size is not None:
        half_size = max_size // 2
        non_repeats = non_repeats[:half_size]
        repeats = repeats[:half_size]

    # Label 0 = non-repeat, 1 = repeat
    data = [(seq, 0) for seq in non_repeats] + [(seq, 1) for seq in repeats]
    dataset = pd.DataFrame(data, columns=['sequence', 'label'])

    dataset.to_csv('dna_dataset_full.csv', index=False)
    return dataset


# Compute evaluation metrics using sklearn
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(logits, axis=-1)
    labels = labels.flatten() if len(labels.shape) > 1 else labels  # Ensure labels are 1D

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "matthews_correlation": matthews_corrcoef(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }


# Wrapper function to use with Hugging Face Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


# Trainer wrapper class to manage model training, evaluation, and prediction
class TandemTrainer:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

        self.train_dataset = None  # Run preprocess() to set
        self.test_dataset = None  # Run preprocess() to set
        self.val_dataset = None  # Run preprocess() to set

    # Preprocess the dataset: tokenize sequences and split into train/val/test
    def preprocess(self):
        def preprocess_data(df):
            tokenized = self.tokenizer(df["sequence"], padding="max_length", truncation=True, return_tensors="pt")
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": torch.tensor(df["label"], dtype=torch.long),
            }

        # Split dataset: 80% train, 20% test
        train_df, test_df = train_test_split(self.dataset, test_size=0.2, random_state=42)

        # Further split train into train (80%) and validation (20%) of original training data
        train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

        # Print dataset sizes
        print(f"Train: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")

        # Convert to Hugging Face Datasets and tokenize
        self.train_dataset = Dataset.from_pandas(train_df)
        self.train_dataset = self.train_dataset.map(preprocess_data, batched=True, batch_size=100, keep_in_memory=False)

        self.test_dataset = Dataset.from_pandas(test_df)
        self.test_dataset = self.test_dataset.map(preprocess_data, batched=True)

        self.val_dataset = Dataset.from_pandas(val_df)
        self.val_dataset = self.val_dataset.map(preprocess_data, batched=True)

    # Train the model using Hugging Face Trainer
    def train(self, training_args=TrainingArguments(
        seed=42,
        output_dir="./results",
        logging_strategy="epoch",
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        learning_rate=0.000001,
        num_train_epochs=10,
    )):
        trainer = transformers.Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            args=training_args,
        )
        trainer.train()

    # Predict class labels (0 or 1) for given sequences
    def predict(self, sequences):
        self.model.eval()  # Switch to evaluation mode

        # Tokenize input sequences
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        # Move inputs to model's device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        return predictions.tolist()

    # Save the trained model and tokenizer
    def save_model(self, path="./results"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    # Load a trained model from disk
    def load_trained_model(self, path="./results"):
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)


# Main function: orchestrates training or loading model and running predictions
def main():
    set_cuda()

    max_size = 3000  # Limit dataset for training; change as needed
    output_dir = "./results"

    # Try to load dataset if it already exists
    try:
        dataset = pd.read_csv("dna_dataset.csv")
    except FileNotFoundError:
        # If not, regenerate from source files
        non_repeats = get_non_repeats("output.txt")
        repeats = get_repeats("VNTRseek_NA19240_ERX283215.vcf")
        dataset = make_dataset(non_repeats, repeats, max_size=max_size)
        dataset.to_csv("dna_dataset.csv")

    # Load existing model if saved, else initialize a new one
    if os.path.exists(os.path.join(output_dir, "model.safetensors")):
        print("Loading previously trained model...")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    else:
        print("Training new model...")
        tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")
        model = AutoModelForSequenceClassification.from_pretrained("PoetschLab/GROVER")

    # Create trainer object
    tt = TandemTrainer(model=model, tokenizer=tokenizer, dataset=dataset)

    # Train the model if no pretrained weights found
    if not os.path.exists(os.path.join(output_dir, "model.safetensors")):
        tt.preprocess()
        tt.train()
        tt.save_model()

    sample_sequences = ["ACTGGAGGGTCTGCAGGCAGGTACCTGGGTGCTGGAGGGTCTGCAGGCAGGTACCTGGG",
                        "TCTCTACAGCGTGTGGACAGCGCGCGCCCTCTACAGCGTGTGGGTGCGCGTGCTCTCTACAGCGTGTGGATGCGTGTGCCCTCTACAGTGTGTGGATGTGTGTGC",
                        "TTATTACATATGGTTTATTACCATATGTAATATTACATATTACATATGGTTTATTACCATATGTAATATTACATATTACATATGGTTTATTACCATATGTAATATTACATATTACATATGGTTTATTACCATATGTAATATTACATATTACATATGGTTTATTACATAT",
                        "TTTGGGCCC",
                        "GGGTGTGGGGTAGCCCACCTACAGTACCATGCATATATATTATAAATTTTTAAACTGTGTGCACTTATGCTAACTGCTGGAAGGTTTAATTGCTTCCTTGATATTTTCAGCATGTCAGAAATCTCCACCTCCTAGTCCTTAGATTTTAGATTGTAACAAATTGCAGCCAAACAGCTAAGGTTTGAATTCTTATTCTTTATTTATATATGTTATAGATTTTACTGTCAATAAAACTAGGATAATT",
                        "AGATCATCTTCAAGCTGTATTTTTGTATCTACATATAGTGACCTTCAAAGGGATTCATTATATTTCATCTTGTGTATTGTATGTTTGACAAGTATATTTGTTTTGATGGTTCACTGATTTCAGTCCTTGGTATTTACCTCAAGTTCATTCTGGGATCTGGTTCTATTCATACACATTATCATAGCTATTTGGAAGCTTACTTGAAATCTTCTCTTTGAAAGAAATATCAGCCTATCTCAGGTGTGAGCTTCAGTGTAACATTGATGTGAGTGAAATAGTATGTTATATAGAAAATTTTATTTTCCCCTGCCATCCCTTACATAAGATTTTTTAAATACCTGAAAGAAAACCTTAGAGCTCTTCATTCAGTCTTTGGAAAAGCCTGATGTGAACCATCATATGGGAGGAATCAGCTGCATCTGTCATTCTTTCTTCCCATGAGTTCCCATAATTTCATTTTCTTAATATGTAGTATAGGGTGGATAGCAGTCATAATCCCCTTTTCCATCTCAAGCCAGCCCACACTCCTTAGCCTCATGCACATGACATTCCTCTTAAGTTGTGATCTGGAGAAGGCTGGAGATGTTTGCT",]
    predicted_classes = tt.predict(sample_sequences)
    print("Predictions:", predicted_classes)


if __name__ == "__main__":
    main()
