from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler,
)
from datasets import load_from_disk
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm import tqdm

accelerate = Accelerator()

dataset_new = load_from_disk("drug_my_datasets")
list_condition = []
list_result = [list_condition.extend(dataset_new[x]["condition"]) for x in dataset_new]
unique_condition = set(list_condition)
sorted_list_condition = sorted(list(unique_condition))
labels_condition = {value: key for key, value in enumerate(sorted_list_condition)}

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=len(unique_condition),
    id2label={i: c for i, c in enumerate(sorted_list_condition)},
    label2id={c: i for i, c in enumerate(sorted_list_condition)},
)


def tokenizer_function(example):
    result = tokenizer(
        example["review"],
        example["condition"],
        truncation=True,
        max_length=128,
    )
    return result


def labels_function(example):
    example["labels"] = labels_condition[example["condition"]]
    return example


tokenizer_datasets = dataset_new.map(tokenizer_function, batched=True)
tokenizer_datasets = tokenizer_datasets.map(labels_function)

tokenizer_datasets = tokenizer_datasets.remove_columns(
    [
        "patient_id",
        "drugName",
        "rating",
        "date",
        "usefulCount",
        "review_length",
        "condition",
        "review",
    ]
)

tokenizer_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    tokenizer_datasets["train"], batch_size=8, shuffle=True, collate_fn=data_collator
)
eval_loader = DataLoader(
    tokenizer_datasets["validation"], batch_size=8, collate_fn=data_collator
)

optimizer = AdamW(model.parameters(), lr=5e-5)

train_loader, eval_loader, model, optimizer = accelerate.prepare(
    train_loader, eval_loader, model, optimizer
)

num_epoch = 3
num_step_training = num_epoch * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_step_training,
)

progress_bar = tqdm(range(num_step_training))

model.train()
for epoch in range(num_epoch):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerate.backward(loss=loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

unwrap_model = accelerate.unwrap_model(model=model)

dir_save = "drug-accelerate-final"
unwrap_model.save_pretrained(dir_save)
tokenizer.save_pretrained(dir_save)
