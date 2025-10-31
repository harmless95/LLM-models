from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    revision="d7645e1",
    device=-1,
)
# sequences - Текст для определения к какой метки относится
# candidate_labels - Метки
result = classifier(
    "There are 200 students studying in Russia.",
    candidate_labels=["education", "politics", "business", "country"],
)
print(result)
