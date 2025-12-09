from transformers import AutoTokenizer
from datasets import load_dataset
import time


file_datasets = {
    "train": "./drug_review_datasets/train/0000.parquet",
    "test": "./drug_review_datasets/test/0000.parquet",
    "validation": "./drug_review_datasets/validation/0000.parquet",
}
drug_review_datasets = load_dataset("parquet", data_files=file_datasets)

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenized_function(example):
    return tokenizer(example["review"], truncation=True)


start = time.time()
tokenize_datasets = drug_review_datasets.map(tokenized_function, batched=True)
print("Время работы: ", time.time() - start)
