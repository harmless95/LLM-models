from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)

response_text = ner("My name is Vitaly and I go to school.")

for response in response_text:
    print(response)
print(ner.model.config.id2label)
