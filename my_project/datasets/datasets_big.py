import psutil
from datasets import load_dataset
import timeit

file_datasets = {
    "train": "drug-reviews-train.jsonl",
    "test": "drug-reviews-test.jsonl",
    "validation": "drug-reviews-validation.jsonl",
}
drug_datasets = load_dataset("json", data_files=file_datasets)

total_size = sum(dataset.num_rows for dataset in drug_datasets.values())

size_gb = total_size / (1024**2)


code_snippet = """batch_size = 1000
train_data = drug_datasets['train']
for i in range(0, len(train_data), batch_size):
    batch_indices = range(i, min(i + batch_size, len(train_data)))
    _ = train_data.select(batch_indices)
"""

time = timeit.timeit(stmt=code_snippet, number=1, globals=globals())
print(
    f"Iterated over {len(drug_datasets["train"])} examples (about {size_gb:.1f} MB) in "
    f"{time:.1f}s, i.e. {size_gb/time:.3f} MB/s"
)
