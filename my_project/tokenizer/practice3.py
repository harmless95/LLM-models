from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForTokenClassification.from_pretrained(checkpoint)
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."

inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)

# print(inputs["input_ids"].shape)
# print(outputs.logits.shape)

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()


result = []
inputs_with_offset = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offset.tokens()
offset = inputs_with_offset["offset_mapping"]

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offset[idx]
        result.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )
for item in result:
    print(item)
