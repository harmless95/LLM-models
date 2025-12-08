from transformers import pipeline
import torch

print(torch.cuda.is_available())  # True, если GPU доступен
print(torch.version.cuda)  # Должно показать 12.8
print(torch.__version__)  # 2.7.1+cu128

classifier = pipeline(
    "zero-shot-classification",
    model="sberbank-ai/ruRoberta-large",
    device=0,
)
# sequences - Текст для определения к какой метки относится
# candidate_labels - Метки
result = classifier(
    "В России учатся 200 студентов",
    candidate_labels=["образование", "политика", "бизнес", "страна"],
)
print(result)
