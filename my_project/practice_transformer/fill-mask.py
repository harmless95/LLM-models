from transformers import pipeline

unmasker = pipeline(
    "fill-mask",
    model="distilbert/distilroberta-base",
    revision="fb53ab8",
    device=0,
)

result = unmasker("Today the weather is <mask> and it is raining heavily.", top_k=2)
print(result)
