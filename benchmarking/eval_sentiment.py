import json
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict
import torch
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DownstreamEval")

class DownstreamBenchmarker:
    """
    Evaluates a tokenizer on a mock Sentiment Analysis task.
    In a real publication, this would use AI4Bharat datasets.
    For this 'Scientific Proof' run, we use a synthetic Tamil sentiment dataset
    to verify if AMB's morphological clarity helps classification.
    """
    def __init__(self, tokenizer_path: str, device: str = "cpu"):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.device = device

    def create_synthetic_sentiment_data(self):
        """Creates a small Tamil sentiment dataset."""
        data = [
            {"text": "மிகவும் அருமையான படம்", "label": 1}, # Very good movie
            {"text": "மோசமான அனுபவம்", "label": 0},      # bad experience
            {"text": "இந்த புத்தகம் எனக்கு பிடித்திருக்கிறது", "label": 1}, # I like this book
            {"text": "நேரம் வீணாகிப்போனது", "label": 0},   # time wasted
            {"text": "அற்புதமான நடிப்பு", "label": 1},      # wonderful acting
            {"text": "பிடிக்கவில்லை", "label": 0},          # did not like
            {"text": "சிறந்த சேவை", "label": 1},            # excellent service
            {"text": "வேலை செய்யவில்லை", "label": 0}        # not working
        ] * 10 # Repeat to make it trainable for a few steps
        
        with open("sentiment_data.jsonl", "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        return "sentiment_data.jsonl"

    def run_sentiment_task(self):
        log.info("Starting Sentiment Analysis Downstream Task...")
        data_path = self.create_synthetic_sentiment_data()
        dataset = load_dataset("json", data_files=data_path, split="train").train_test_split(test_size=0.2)

        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=32)

        tokenized_datasets = dataset.map(preprocess_function, batched=True)

        # We use a tiny BERT-layer for the test
        from transformers import BertConfig, BertForSequenceClassification
        config = BertConfig(
            vocab_size=len(self.tokenizer),
            num_labels=2,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256
        )
        model = BertForSequenceClassification(config)

        training_args = TrainingArguments(
            output_dir="./results_sentiment",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            remove_unused_columns=False,
            logging_steps=10,
            report_to="none",
            no_cuda=(self.device == "cpu")
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"]
        )

        trainer.train()
        results = trainer.evaluate()
        log.info(f"Sentiment Analysis Results: {results}")
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    bench = DownstreamBenchmarker(args.tokenizer, args.device)
    bench.run_sentiment_task()
