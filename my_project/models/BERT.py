from transformers import BertConfig, BertModel

config = BertConfig()

model = BertModel(config)

print(config)
