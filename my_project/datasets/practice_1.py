from datasets import load_dataset

# способ загрузки датасетов, которые не размещены на Hugging Face Hub
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
datasets_date = load_dataset("json", data_files=data_files, field="data")

print(datasets_date)
