from transformers import pipeline

unmask = pipeline("zero-shot-classification")

# Определяет предложение к какой теме относится
result = unmask(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print(result)
