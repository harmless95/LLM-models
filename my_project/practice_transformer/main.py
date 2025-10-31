from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f",
    device=-1,
)

result1 = classifier("I love using Hugging Face models on CPU!")
print(result1)

# Мы можем передать на вход сразу несколько предложений!
result2 = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
print(result2)
