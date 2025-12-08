from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# sequence1_ids = [[200, 200, 200]]
# sequence2_ids = [[200, 200]]
# batched_ids = [
#     [200, 200, 200],
#     [200, 200, tokenizer.pad_token_id],
# ]
# print(model(torch.tensor(sequence1_ids)).logits)
# print(model(torch.tensor(sequence2_ids)).logits)
# print(model(torch.tensor(batched_ids)).logits)

# макска внимания

# batched_ids = [
#     [200, 200, 200],
#     [200, 200, tokenizer.pad_token_id],
# ]
#
# attention_mask = [
#     [1, 1, 1],
#     [1, 1, 0],
# ]
#
# outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
# print(outputs.logits)

sequence = [
    "I’ve been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
tokens = tokenizer(sequence, padding=True, truncation=True, return_tensors="pt")
print("666", tokens)
output = model(**tokens)
print(output.logits)
