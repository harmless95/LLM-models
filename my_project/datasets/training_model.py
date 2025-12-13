from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_from_disk
import numpy as np
import evaluate

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
        valid_label = [label for label in labels if label >= 0]
        label = 1 if valid_label else 0
        labels_list.append(label)

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
collator_data = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)[
            "accuracy"
        ],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"],
    }


train_args = TrainingArguments(
    "drug-trainer",
    eval_strategy="epoch",
    per_device_train_batch_size=16,  # ✅ x2 скорость
    fp16=True,  # ✅ x2-3 скорость на GPU
)
trainer = Trainer(
    model=model,
    args=train_args,
    data_collator=collator_data,
    train_dataset=tokenizer_datasets["train"],
    eval_dataset=tokenizer_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train(resume_from_checkpoint="drug-trainer/checkpoint-24500")

trainer.save_model("drug-bert-final")
tokenizer.save_pretrained("drug-bert-final")
