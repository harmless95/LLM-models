from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "I hate this so much!"

tokens = tokenizer.tokenize(sequence)
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decode_ids = tokenizer.decode([146, 4819, 1142, 1177, 1277, 106])
print(decode_ids)
