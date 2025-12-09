import time

from datasets import load_dataset
import html

data_files = {
    "train": "./drug_review_datasets/train/0000.parquet",
    "test": "./drug_review_datasets/test/0000.parquet",
    "validation": "./drug_review_datasets/validation/0000.parquet",
}
drug_sample = load_dataset(
    "parquet",
    data_files=data_files,
)
# drug_sample = data_datasets["train"].shuffle(seed=42).select(range(1000))

# for split in drug_sample.keys():
#     assert len(drug_sample[split]) == len(drug_sample[split].unique("patient_id"))


def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


# используем лямбду для проверки, что бы при обработки в нижний регистр значение небыло None
drug_sample = drug_sample.filter(lambda x: x["condition"] is not None)
drug_sample.map(lowercase_condition)

# print(drug_sample["train"]["condition"][:3])


# создаем новый столбец размер которого зависит от кол-во слов в отзывах
def compute_review_length(example):
    return {"length_review": len(example["review"].split())}


drug_sample = drug_sample.map(compute_review_length)
##  Альтернативный вариант добавления нового столбца в датасет – использовать функцию Dataset.add_column().
## Она позволяет создать новый столбец из Python-списка или NumPy-массива, что может быть удобно, если функция Dataset.map() не очень подходит для вашего случая.

# print(drug_sample["train"][0])
# print(drug_sample["train"].sort("length_review")[-3:])
#
# drug_sample = drug_sample.filter(lambda x: x["length_review"] > 30)
#
# print(drug_sample.num_rows)
start = time.time()
# Избавляемся от html кода в отзывах
# drug_sample = drug_sample.map(lambda x: {"review": html.unescape(x["review"])})
# использует batched и сильно быстрее работает чем предыдущая

new_drug_datasets = drug_sample.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)
print("Время работы: ", time.time() - start)
