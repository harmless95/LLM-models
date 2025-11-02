from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# последоваетльность из 1
# sequence = "I've been waiting for a HuggingFace course my whole life."
# последовательность из нескольких
# sequence = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Здесь переменная model_inputs содержит все, что необходимо для нормальной работы модели
# model_inputs = tokenizer(sequence)

# Дополнение последовательностей до максимальной длины последовательности
# model_inputs = tokenizer(sequence, padding="longest")

# Дополнение последовательностей до максимальной длины модели
# (512 для BERT или DistilBERT)
# model_inputs = tokenizer(sequence, padding="max_length")

# Дополнение последовательностей до заданной максимальной длины если длина больше заданной она не меняется
# Если добавить truncation=True то урежет если длина больше или добавит если меньше
# model_inputs = tokenizer(sequence, padding="max_length", max_length=8, truncation=True)
# print(model_inputs)

# Разновидность тензеров
# model_inputs_pt = tokenizer(sequence, padding=True, return_tensors="pt")
# print("Model PyTorch: ", model_inputs_pt)

# model_inputs_tf = tokenizer(sequence, padding=True, return_tensors="tf")
# print("Model TensorFlow: ", model_inputs_tf)
#
# model_inputs_np = tokenizer(sequence, padding=True, return_tensors="np")
# print("Model NumPy: ", model_inputs_np)

# Специальные токены
sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
# Добавлен в начало и конец
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# Токенизатор добавил специальное слово [CLS] в начале и специальное слово [SEP] в конце.
model_decode_1 = tokenizer.decode(model_inputs["input_ids"])
print(model_decode_1)
model_decode_2 = tokenizer.decode(ids)
print(model_decode_2)
