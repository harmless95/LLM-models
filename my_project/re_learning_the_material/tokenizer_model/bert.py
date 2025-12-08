from transformers import AutoTokenizer
import torch

checkpoint = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
decode_tokenize = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decode_tokenize)
