from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(
        example["sentence1"], example["sentence2"], truncation=True
    )  # берем из данных 2 предложения и производим усечение


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
samples = tokenized_datasets["train"][:8]
samples = {
    k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]
}


batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})
