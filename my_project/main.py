from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
import torch

# Преобразование предложения в токены
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sentence = "Transformers are amazing!"
tokens = tokenizer.tokenize(sentence)
print(f"Tokens: {tokens}")

# Подготовка данных к обучению
# Создаем тренировочные данные с помощью маски
masked_sentence = "Transformers are [MASK]!"
input_ids = tokenizer.encode(masked_sentence, return_tensors="pt")

# Создаем метки, заменяя все не замаскированные токены на -100
labels = input_ids.clone()
mask_token_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
labels[input_ids != tokenizer.mask_token_id] = -100

# Пример создания кастомного набора данных
# Здесь мы используем один пример для иллюстрации
train_dataset = Dataset.from_dict(
    {"input_ids": [input_ids[0].tolist()], "labels": [labels[0].tolist()]}
)

# Загружаем предварительно обученную модель BERT
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Выставляем параметры обучения
training_args = TrainingArguments(
    output_dir="./results",  # папка для выходных данных
    learning_rate=5e-5,  # темп обучения
    per_device_train_batch_size=8,  # размер батча
    num_train_epochs=3,  # количество эпох
    logging_dir="./logs",  # папка для логов
)

# Создаем обучающий объект
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Обучаем модель
trainer.train()

# Сохраняем обученную модель
model.save_pretrained("./my_bert_model")
