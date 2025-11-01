from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# print(model)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

outputs = model(**inputs)

# print(outputs.logits)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
# print(predictions)

res = model.config.id2label
print(res)
