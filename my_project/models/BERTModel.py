from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
model.save_pretrained("directory_on_my_computer")
