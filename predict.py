# predict.py
from transformers import pipeline

classifier = pipeline("text-classification", model="./law-model", tokenizer="./law-model")
result = classifier("What is the punishment for theft?")
print(result)
