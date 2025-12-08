from transformers import pipeline, set_seed

generator = pipeline(
    "text-generation",
    model="distilgpt2",
    # revision="d7645e1",
    device_map="auto",
)
set_seed(42)
result = generator(
    "When will I be hired as a Python developer?",
    max_new_tokens=30,  # длина именно продолжения
    num_return_sequences=2,
    do_sample=True,  # стохастическая генерация
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)
for response_text in result:
    print("----")
    print(response_text)
