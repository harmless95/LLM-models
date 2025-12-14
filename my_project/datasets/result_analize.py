from transformers import pipeline

# Твои реальные лейблы (из config, начиная с 73)
MEDICAL_LABELS = [
    "birth control",
    "depression",
    "pain",
    "weight loss",
    "anxiety",
    "insomnia",
    "acne",
    "migraine",
    "hot flashes",
    "crohn's disease",
    "diabetes type 2",
    "high blood pressure",
    "copd",
    "schizophrenia",
]

classifier = pipeline(
    "zero-shot-classification",
    model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    candidate_labels=MEDICAL_LABELS,
)

review = """I was on this for 5 years (and birth control pills...)"""
result = classifier(review, multi_label=True)

print("Top-5:", result["labels"][:5])
print("Scores:", [f"{s:.3f}" for s in result["scores"][:5]])
