from transformers import AutoTokenizer, pipeline

token_classifier = pipeline(
    "token-classification", aggregation_strategy="simple"
)  # если добавить aggregation_strategy то он соберет по словам.
# Значения "simple" оценка является средним значением оценок каждого токена данной сущности

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn"


encoding = tokenizer(example)

# print(encoding.is_fast)  # Проверяем быстрый или медленный tokenizer
# print(encoding.tokens())  # Можно просмотреть токены с (), без будут ID
# print(encoding.word_ids())  # [None, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, None] где одинаковые id это одно слово

# start, end = encoding.word_to_chars(3)
# print(
#     example[start:end]
# )  # Можем по id узнать все слово целиком, так как хранится токенами ('S', '##yl', '##va', '##in')


token_cls = token_classifier(example)
for i in token_cls:
    print(i)
