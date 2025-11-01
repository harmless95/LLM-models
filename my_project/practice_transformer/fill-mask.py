from transformers import pipeline

unmasker = pipeline(
    "fill-mask", model="distilbert/distilroberta-base", revision="fb53ab8", device=0
)

result = unmasker("I got a <mask> grade at school.", top_k=2)
print(result)
