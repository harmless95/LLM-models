from transformers import pipeline

checkpoint = "drug-bert-final"

classifier = pipeline(
    "text-classification",
    model=checkpoint,
    return_all_scores=True,
)
review = "Ditto on rebound sleepless when discontinued. I have done very strange things with no memory including taking additional Ambien. It has helped me sleep when under extreme stress but watch out. Now I am trying to learn how to sleep naturally."
result = classifier(review)
print(result)
