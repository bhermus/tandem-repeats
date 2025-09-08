"""
GROVER DNA Classifier – Student‑Friendly, Heavily Commented Version
=================================================================
What it does (high level)
-------------------------
1) Reads two sources of sequences:
   • non‑repeats from a text file that looks like FASTA/FASTA‑like (headers start with '>').
   • repeats from a VCF file (uses the REF column as a proxy sequence here).
2) Builds a balanced dataset with labels: 0 = non‑repeat, 1 = repeat.
3) Tokenizes with Hugging Face's GROVER tokenizer and trains a classification head on top
   of the GROVER encoder using the HF Trainer.
4) Saves/loads the model and runs a small prediction demo.

Notes for future contributors
-----------------------------
• GROVER (PoetschLab/GROVER) is a masked‑language model for DNA trained with BPE (~600 vocab).
  We use it via AutoTokenizer and AutoModelForSequenceClassification.
• The current VCF reader treats the REF allele as the sequence; in real analyses you may
  want to extract reference context around variants (requires a FASTA + coordinates).
• Long sequences are truncated to the model max length (default ~512 for many encoders).
  Consider sliding windows if your sequences are very long.
• For imbalanced classes, consider class weights, focal loss, or stratified splits.

Dependencies
------------
- transformers, datasets, scikit‑learn, torch, numpy, pandas, matplotlib

Run
---
python tandem-repeats-heavy-comments.py

"""
from __future__ import annotations

# --- Core libs ---
import os
import re
import json
from typing import List, Tuple

# --- Numerics / ML ---
import numpy as np
import pandas as pd
import torch

# --- Plotting (not strictly needed for training, but kept from original) ---
import matplotlib.pyplot as plt

# --- Metrics & split ---
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

# --- HF Transformers / Datasets ---
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from datasets import Dataset

# Pretty‑print options for pandas (as in original)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


# ===============================================================
# Hardware selection / diagnostics
# ===============================================================
def set_cuda() -> torch.device:
    """Print basic GPU info and return the chosen torch.device.

    Returns
    -------
    torch.device
        'cuda' if available, otherwise 'cpu'.
    """
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)
    if num_gpus > 0:
        try:
            print("GPU Name:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("Could not query GPU name:", e)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


# ===============================================================
# Data loading helpers
# ===============================================================
def get_non_repeats(path_to_file: str) -> List[str]:
    """Extract non‑repeat candidate sequences from a FASTA‑like text file.

    Behavior (kept from original):
    - Reads up to ~30M characters (via readlines(max_lines)).
    - Concatenates all non‑header lines (not starting with '>') into one long string.
    - Splits that long string on runs of 'N' (unknown bases) to produce sub‑sequences.
    - Filters out very short lines (< 26 chars).

    Returns
    -------
    List[str]
        List of sequences interpreted as non‑repeats.
    """
    max_lines = 30_000_000  # soft cap; used with readlines to avoid huge memory spikes

    with open(path_to_file, "r", encoding="utf-8") as f:
        lines = f.readlines(max_lines)

    # Keep only lines that are (a) not FASTA headers and (b) reasonably long
    payload = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s[0] == ">":
            # header line; skip
            continue
        if len(s) <= 25:
            # too short to be informative for the classifier; skip (matches original)
            continue
        payload.append(s)

    # Concatenate remaining lines and split on runs of 'N' (unknown bases)
    data = "".join(payload)
    txt_sequences = re.split(r"N+", data)

    # Show a small sample and counts for quick sanity check (as in original)
    print("[non_repeats] sample:", txt_sequences[:10])
    print("[non_repeats] total:", len(txt_sequences))

    return txt_sequences


def get_repeats(path_to_file: str) -> List[str]:
    """Extract repeat candidate sequences from a VCF by using the REF column as proxy.

    Notes
    -----
    - A real pipeline may need to fetch flanking context from a reference genome.
    - Here we replicate the original behavior: collect column 4 (REF) for up to 30k lines.

    Returns
    -------
    List[str]
        List of sequences interpreted as repeats.
    """
    vcf_sequences: List[str] = []
    max_lines = 30_000

    with open(path_to_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i > max_lines:
                break
            if line.startswith("#"):
                # header / metadata
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) > 3:
                ref = cols[3]
                vcf_sequences.append(ref)

    # Quick diagnostics as in original
    print("[repeats] sample:", vcf_sequences[:10])
    if vcf_sequences:
        print("[repeats] min length:", min(len(seq) for seq in vcf_sequences))
    print("[repeats] total:", len(vcf_sequences))

    return vcf_sequences


# ===============================================================
# Dataset assembly
# ===============================================================
def make_dataset(non_repeats: List[str], repeats: List[str], max_size: int | None = None) -> pd.DataFrame:
    """Create a labeled dataframe with columns ['sequence', 'label'].

    Parameters
    ----------
    non_repeats : List[str]
        Sequences labeled 0.
    repeats : List[str]
        Sequences labeled 1.
    max_size : int | None
        If provided, we take a balanced subset with half from each class.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: sequence (str), label (int).
    """
    if max_size is not None:
        half = max_size // 2
        non_repeats = non_repeats[:half]
        repeats = repeats[:half]

    # Build rows and write a full copy to CSV for traceability
    rows = [(s, 0) for s in non_repeats] + [(s, 1) for s in repeats]
    df = pd.DataFrame(rows, columns=["sequence", "label"])
    df.to_csv("dna_dataset_full.csv", index=False)
    return df


# ===============================================================
# Metrics
# ===============================================================
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Compute standard classification metrics from raw logits.

    We use argmax over the last dimension to get class predictions.
    Metrics are macro‑averaged to avoid overweighting majority classes.
    """
    # Convert logits -> predictions
    preds = np.argmax(logits, axis=-1)

    # Ensure labels are 1D
    labels = labels.flatten() if labels.ndim > 1 else labels

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "matthews_correlation": matthews_corrcoef(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


def compute_metrics(eval_pred) -> dict:
    """HF Trainer hook to compute metrics during eval.

    `eval_pred` can be a tuple (logits, labels) or similar. We normalize to logits np.array.
    """
    logits, labels = eval_pred
    # Some models return a tuple; we take the first element as logits.
    if isinstance(logits, tuple):
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


# ===============================================================
# Training wrapper
# ===============================================================
class TandemTrainer:
    """Wraps tokenization, splitting, training, evaluation, and prediction.

    Attributes
    ----------
    model : PreTrainedModel
        HF sequence classification model (initialized from GROVER or a saved dir).
    tokenizer : PreTrainedTokenizer
        GROVER tokenizer (AutoTokenizer.from_pretrained("PoetschLab/GROVER") or a saved dir).
    dataset : pd.DataFrame
        DataFrame with columns 'sequence' and 'label'.

    Datasets (populated after preprocess):
    -------------------------------------
    train_dataset, val_dataset, test_dataset : datasets.Dataset
    """

    def __init__(self, model, tokenizer, dataset: pd.DataFrame):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def preprocess(self) -> None:
        """Tokenize text and build HF Datasets for train/val/test.

        This mirrors your original logic with two changes worth noting:
        - We keep the same (80/20) split then (within train) a further 75/25 to produce
          an overall 60/20/20 split (train/val/test). This matches the original math.
        - We *could* stratify by label to preserve class ratios; left as a TODO if needed.
        """
        def preprocess_batch(batch_df: pd.DataFrame) -> dict:
            # The tokenizer returns PyTorch tensors when return_tensors='pt'. Datasets can
            # store them, but returning lists can be more interoperable. We keep your
            # original behavior (tensors) for fidelity.
            tok = self.tokenizer(
                batch_df["sequence"].tolist(),
                padding="max_length",          # fixed‑size padding (simple, but may waste space)
                truncation=True,                # truncate long sequences to model max length
                return_tensors="pt",
            )
            return {
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "labels": torch.tensor(batch_df["label"].values, dtype=torch.long),
            }

        # Primary split: train (80%), test (20%)
        train_df, test_df = train_test_split(self.dataset, test_size=0.2, random_state=42)

        # Secondary split: val is 25% of the train portion -> overall 20%
        train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

        print(f"Train: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")

        # Wrap with HF Datasets and apply tokenization map
        self.train_dataset = Dataset.from_pandas(train_df)
        self.train_dataset = self.train_dataset.map(preprocess_batch, batched=True, batch_size=100, keep_in_memory=False)

        self.val_dataset = Dataset.from_pandas(val_df)
        self.val_dataset = self.val_dataset.map(preprocess_batch, batched=True)

        self.test_dataset = Dataset.from_pandas(test_df)
        self.test_dataset = self.test_dataset.map(preprocess_batch, batched=True)

    def train(self, training_args: TrainingArguments | None = None) -> None:
        """Run HF Trainer training loop.

        Parameters
        ----------
        training_args : TrainingArguments | None
            If None, we build a default set (mirrors your original values).
        """
        if training_args is None:
            training_args = TrainingArguments(
                seed=42,
                output_dir="./results",
                logging_strategy="epoch",
                per_device_train_batch_size=16,
                evaluation_strategy="epoch",
                learning_rate=1e-6,   # very conservative LR; may want to tune (1e-5/2e-5)
                num_train_epochs=10,
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
            )

        trainer = transformers.Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            args=training_args,
        )
        trainer.train()

    def predict(self, sequences: List[str]) -> List[int]:
        """Predict class indices for a list of raw DNA strings.

        Returns
        -------
        List[int]
            Argmax class ids per input sequence.
        """
        self.model.eval()

        # Tokenize to the model's max length
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        # Send tensors to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

        return preds.tolist()

    def save_model(self, path: str = "./results") -> None:
        """Save model + tokenizer to a directory (for later reuse)."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_trained_model(self, path: str = "./results") -> None:
        """Load a previously saved model + tokenizer from disk.

        Note: replaces in‑memory model/tokenizer with loaded versions.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)


# ===============================================================
# Main entry point
# ===============================================================
def main() -> None:
    # 1) Device info (prints GPU if available)
    device = set_cuda()

    # 2) Configuration / paths
    max_size = 3000                 # cap dataset size (balanced) for quick experiments
    output_dir = "./results"        # where to save/load model

    # 3) Load or build dataset
    try:
        # Prefer an existing CSV if present (fast reloads)
        dataset = pd.read_csv("dna_dataset.csv")
        assert {"sequence", "label"} <= set(dataset.columns)
    except (FileNotFoundError, AssertionError):
        # Rebuild from raw sources
        non_repeats = get_non_repeats("output.txt")
        repeats = get_repeats("VNTRseek_NA19240_ERX283215.vcf")
        dataset = make_dataset(non_repeats, repeats, max_size=max_size)
        dataset.to_csv("dna_dataset.csv", index=False)

    # 4) Load a saved model if present; otherwise start from GROVER checkpoint
    model_path = os.path.join(output_dir, "model.safetensors")
    if os.path.exists(model_path):
        print("Loading previously trained model from", output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    else:
        print("Training new model from PoetschLab/GROVER …")
        tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")
        model = AutoModelForSequenceClassification.from_pretrained("PoetschLab/GROVER")

    # Move model to device if using manual loops later; HF Trainer will manage device itself
    model.to(device)

    # 5) Build trainer and (optionally) train
    tt = TandemTrainer(model=model, tokenizer=tokenizer, dataset=dataset)

    if not os.path.exists(model_path):
        tt.preprocess()
        tt.train()
        tt.save_model(output_dir)

    # 6) Small demo predictions
    sample_sequences = [
        "ACTGGAGGGTCTGCAGGCAGGTACCTGGGTGCTGGAGGGTCTGCAGGCAGGTACCTGGG",
        "TCTCTACAGCGTGTGGACAGCGCGCGCCCTCTACAGCGTGTGGGTGCGCGTGCTCTCTACAGCGTGTGGATGCGTGTGCCCTCTACAGTGTGTGGATGTGTGTGC",
        (
            "TTATTACATATGGTTTATTACCATATGTAATATTACATATTACATATGGTTTATTACCATATGTAATATTACATATTACATATGGTTTATTACCATATGTAAT"
            "ATTACATATTACATATGGTTTATTACCATATGTAATATTACATATTACATATGGTTTATTACATAT"
        ),
        "TTTGGGCCC",
        (
            "GGGTGTGGGGTAGCCCACCTACAGTACCATGCATATATATTATAAATTTTTAAACTGTGTGCACTTATGCTAACTGCTGGAAGGTTTAATTGCTTCCTTGATAT"
            "TTTCAGCATGTCAGAAATCTCCACCTCCTAGTCCTTAGATTTTAGATTGTAACAAATTGCAGCCAAACAGCTAAGGTTTGAATTCTTATTCTTTATTTATATAT"
            "GTTATAGATTTTACTGTCAATAAAACTAGGATAATT"
        ),
        (
            "AGATCATCTTCAAGCTGTATTTTTGTATCTACATATAGTGACCTTCAAAGGGATTCATTATATTTCATCTTGTGTATTGTATGTTTGACAAGTATATTTGTTTT"
            "GATGGTTCACTGATTTCAGTCCTTGGTATTTACCTCAAGTTCATTCTGGGATCTGGTTCTATTCATACACATTATCATAGCTATTTGGAAGCTTACTTGAAATCT"
            "TCTCTTTGAAAGAAATATCAGCCTATCTCAGGTGTGAGCTTCAGTGTAACATTGATGTGAGTGAAATAGTATGTTATATAGAAAATTTTATTTTCCCCTGCCATC"
            "CCTTACATAAGATTTTTTAAATACCTGAAAGAAAACCTTAGAGCTCTTCATTCAGTCTTTGGAAAAGCCTGATGTGAACCATCATATGGGAGGAATCAGCTGCAT"
            "CTGTCATTCTTTCTTCCCATGAGTTCCCATAATTTCATTTTCTTAATATGTAGTATAGGGTGGATAGCAGTCATAATCCCCTTTTCCATCTCAAGCCAGCCCACA"
            "CTCCTTAGCCTCATGCACATGACATTCCTCTTAAGTTGTGATCTGGAGAAGGCTGGAGATGTTTGCT"
        ),
    ]
    preds = tt.predict(sample_sequences)
    print("Predictions:", preds)


if __name__ == "__main__":
    main()
