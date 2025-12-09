from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
import evaluate
from accelerate import Accelerator

accelerate = Accelerator()

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


def tokenizer_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


data_datasets = raw_datasets.map(tokenizer_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = data_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    outputs = model(**batch)
    break

optim = AdamW(model.parameters(), lr=3e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optim, num_warmup_steps=0, num_training_steps=num_training_steps
)

train_dataloader, eval_dataloader, model, optim, lr_scheduler = accelerate.prepare(
    train_dataloader, eval_dataloader, model, optim, lr_scheduler
)

progress_bar = tqdm(range(num_training_steps))
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        optim.zero_grad()
        with accelerate.autocast():
            outputs = model(**batch)
            loss = outputs.loss
        accelerate.backward(loss)

        optim.step()
        lr_scheduler.step()
        progress_bar.update(1)

metric_evaluate = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric_evaluate.add_batch(predictions=predictions, references=batch["labels"])

print(metric_evaluate.compute())
