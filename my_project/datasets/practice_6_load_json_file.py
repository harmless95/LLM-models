from datasets import load_dataset

file_datasets = {
    "train": "drug-reviews-train.jsonl",
    "test": "drug-reviews-test.jsonl",
    "validation": "drug-reviews-validation.jsonl",
}
drud_datasets = load_dataset("json", data_files=file_datasets)
