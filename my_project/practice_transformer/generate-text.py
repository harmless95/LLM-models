from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2",
    # revision="d7645e1",
    device=-1,
    num_return_sequences=2,
    max_length=15,
)

result = generator("When will I be hired as a Python developer?")
print(result)
