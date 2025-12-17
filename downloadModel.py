from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "gdewael/MaldiTransformer"

# This will use your cached Hugging Face credentials from huggingface-cli login
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./maldi_model")
model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir="./maldi_model")

print("Model and tokenizer downloaded to ./maldi_model")

