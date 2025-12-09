from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

file_datasets = {
    "train": "./drug_review_datasets/train/0000.parquet",
    "test": "./drug_review_datasets/test/0000.parquet",
    "validation": "./drug_review_datasets/validation/0000.parquet",
}
drug_datasets = load_dataset("parquet", data_files=file_datasets)

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenized_function(example):
    result = tokenizer(
        example["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    sample_map = result.pop("overflow_to_sample_mapping")
    for k, v in example.items():
        result[k] = [v[x] for x in sample_map]
    return result


tokenized_datasets = drug_datasets.map(tokenized_function, batched=True)
# result = tokenized_function(drug_datasets["train"][0])
# result = [len(inp) for inp in result["input_ids"]]
drug_datasets.set_format("pandas")

train_df = drug_datasets["train"][:]
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "count": "frequency"})
)

freq_datasets = Dataset.from_pandas(frequencies)
print(freq_datasets)
drug_datasets.reset_format()
