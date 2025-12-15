from huggingface_hub import hf_hub_url
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

checkpoint = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
datasets_file = hf_hub_url(
    repo_id="lewtun/github-issues",
    filename="datasets-issues-with-comments.jsonl",
    repo_type="dataset",
)

model = AutoModel.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
dataset_hf = load_dataset("json", data_files=datasets_file, split="train")
device = torch.device("cuda")
model.to(device)

dataset_hf = dataset_hf.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)

columns = dataset_hf.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(
    columns
)  # Находит в dataset все поля которых нет в columns_to_leep
dataset_hf = dataset_hf.remove_columns(columns_to_remove)  # Удаляет не нужные поля

dataset_hf.set_format("pandas")
copy_dataset = dataset_hf[:]

comments_dataset = copy_dataset.explode("comments", ignore_index=True)
comments_dataset = Dataset.from_pandas(comments_dataset)

comments_dataset = comments_dataset.map(
    lambda x: {"length_comments": len(x["comments"].split())}
)
comments_dataset = comments_dataset.filter(lambda x: x["length_comments"] > 15)


def concatenate_function(example):
    return {
        "text": example["title"] + " \n" + example["body"] + " \n" + example["comments"]
    }


comments_dataset = comments_dataset.map(concatenate_function)


def cls_pooling(example_model):
    return example_model.last_hidden_state[:, 0]


def get_embeddings(text_list):
    encoding_input = tokenizer(
        text_list,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    encoding_input = {key: value.to(device) for key, value in encoding_input.items()}
    model_output = model(**encoding_input)
    return cls_pooling(model_output)


embedding = get_embeddings(comments_dataset["text"][0])
embeddings_datasets = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

embeddings_datasets.add_faiss_index(column="embeddings")


question = "How can I load a dataset offline?"
question_embeddings = get_embeddings([question]).cpu().detach().numpy()

score, samples = embeddings_datasets.get_nearest_examples(
    "embeddings", question_embeddings, k=5
)


samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = score
samples_df.sort_values("scores", ascending=False, inplace=True)

for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()
