from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка предобученной модели и токенизатора
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Вводный текст
input_text = "Привет, как работает большая языковая модель?"

# Преобразование текста в токены
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Генерация продолжения текста
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Декодирование сгенерированных токенов обратно в текст
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
