from transformers import pipeline

# Анализирует позитивный или негативный тон
unmask = pipeline("sentiment-analysis")
result = unmask(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
print(result)
