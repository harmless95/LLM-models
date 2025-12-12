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
        max_length=128,  # если токенов больше 128 режет несколько кусочков
        return_overflowing_tokens=True,  # что бы понимать когда режет какой чанк к какому отзыву относится
    )
    # индекс чанков
    sample_map = result.pop("overflow_to_sample_mapping")
    for k, v in example.items():
        result[k] = [
            v[x] for x in sample_map
        ]  # указываем метке к какому данным(тексту) он относится
    return result


tokenized_datasets = drug_datasets.map(tokenized_function, batched=True)
# result = tokenized_function(drug_datasets["train"][0])
# result = [len(inp) for inp in result["input_ids"]]
# можно поменять формат
# drug_datasets.set_format("pandas")
#
# train_df = drug_datasets["train"][:]
# frequencies = (
#     train_df["condition"]
#     .value_counts()
#     .to_frame()
#     .reset_index()
#     .rename(columns={"index": "condition", "count": "frequency"})
# )
#
# freq_datasets = Dataset.from_pandas(frequencies)
# print(freq_datasets)
# drug_datasets.reset_format()

# для тестирования лучше создавать новый datasets
## После применения train_test_split он разделять данные на 2 части и дает им названия train(80%) и test(20%)
datasets_new = drug_datasets["train"].train_test_split(train_size=0.8, seed=42)
## меняет название
datasets_new["validation"] = datasets_new.pop("test")
## test берется из оригинального datasets
datasets_new["test"] = drug_datasets["test"]

## сохраняет на datasets
datasets_new.save_to_disk("drug_my_datasets")

## создаёт 3 файла в jsonl
for key, value in datasets_new.items():
    value.to_json(f"drug-reviews-{key}.jsonl")
