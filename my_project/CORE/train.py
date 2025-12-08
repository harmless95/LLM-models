from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm

accelerate = Accelerator()
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerate.prepare(
    train_dl, eval_dl, model, optimizer
)

num_epoch = 3
num_training_steps = num_epoch * len(train_dl)
lr_scheduler = get_scheduler(
    "liner",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epoch):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerate.backward(loss=loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grand()
        progress_bar.update(1)
