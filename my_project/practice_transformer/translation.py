from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru", device=0)
result = translator("Hello, how are you?")
print(result)
