from transformers import pipeline

unmask = pipeline("fill-mask", model="DeepPavlov/rubert-base-cased")

response_text = unmask(
    "Урфом хочет [MASK].",
    top_k=3,
)
for response in response_text:
    print(response)
