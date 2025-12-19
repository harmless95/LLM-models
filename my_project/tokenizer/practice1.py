from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("Nan-Do/code-search-net-python")
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")


def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        sample = dataset[start_idx : start_idx + 1000]
        yield sample["original_string"]


training_corpus = get_training_corpus()

new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)


example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias
    """

tokens_tokenizer = new_tokenizer.tokenize(example)
print(tokens_tokenizer)
new_tokenizer.save_pretrained("code-search-net-tokenizer")
