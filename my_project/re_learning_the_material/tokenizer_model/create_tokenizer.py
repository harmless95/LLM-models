from transformers import AutoTokenizer, AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

text_requests = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(text_requests, padding=True, truncation=True, return_tensors="pt")

outputs = model(**inputs)
# Получаем torch.Size([2, 16, 768]) где 2 - число последовательностей, 16 - это длина последовательности, 768 - скрытый размер
print(outputs.last_hidden_state.shape)
