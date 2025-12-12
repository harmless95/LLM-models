from scipy.optimize import bracket
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from datasets import load_from_disk
from torch.utils.data import DataLoader


dataset_file = load_from_disk("drug_my_datasets")

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


def condition_split():
    all_conditions = set()
    for split in dataset_file.keys():
        all_conditions.update(dataset_file[split]["condition"])
    return {c: i for i, c in enumerate(sorted(all_conditions))}


dict_cond = condition_split()


def tokenizer_condition(example):
    labels_list = []

    for condition_str in example["condition"]:
        # разбиваем строку условия на список условий
        conditions = [cond.strip() for cond in condition_str.split(",")]
        # для каждого условия ищем метку
        labels = [dict_cond.get(cond, -1) for cond in conditions]
        labels_list.append(labels)

    # добавляем полученные списки меток в результаты
    example["labels"] = labels_list

    return example


dataset_file = dataset_file.map(tokenizer_condition, batched=True)


def tokenizer_function(example):
    result = tokenizer(
        example["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    sample_map = result.pop("overflow_to_sample_mapping")

    for key, value in example.items():
        result[key] = [value[x] for x in sample_map]

    return result


tokenizer_datasets = dataset_file.map(tokenizer_function, batched=True)
# print(tokenizer_datasets["train"][0])
tokenizer_datasets = tokenizer_datasets.remove_columns(
    ["review_length", "usefulCount", "date", "rating", "drugName"]
)

collator_date = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenizer_datasets["train"], batch_size=8, shuffle=True, collate_fn=collator_date
)
for batch in train_dataloader:
    print({key: value.shape for key, value in batch.items()})
    break
