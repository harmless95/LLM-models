from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# sequence = "I've been waiting for a HuggingFace course my whole life."
#
# tokens = tokenizer.tokenize(sequence)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# Выведет ошибку ожидает последовательность, а мы передает 1 значение
# input_ids = torch.tensor(ids)
# result = model(input_ids)

# Как обойти, что бы работало
# inputs_ids = torch.tensor([ids])

# Батчинг - это отправка нескольких предложений через модель одновременно.
# Если у вас есть только одно предложение, вы можете просто создать батч с одной последовательностью:
# batched_ids = [ids, ids]

# Дополнение (padding), чтобы придать тензорам прямоугольную форму
# padding_id = 100
# batched_ids = [[200, 200, 200], [200, 200, padding_id]]
# inputs_ids = torch.tensor(batched_ids)
# print("Input IDs:", inputs_ids)
#
# output = model(inputs_ids)
# print("Logits:", output.logits)

# sequence1_ids = [[200, 200, 200]]
# sequence2_ids = [[200, 200]]
# batched_ids = [[200, 200, 200], [200, 200, tokenizer.pad_token_id]]
# print(model(torch.tensor(sequence1_ids)).logits)
# print(model(torch.tensor(sequence2_ids)).logits)
# print(model(torch.tensor(batched_ids)).logits)

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

sequence1 = [
    "I’ve been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

inputs = tokenizer(sequence1, padding=True, truncation=True, return_tensors="pt")
output = model(**inputs)
print(output.logits)
