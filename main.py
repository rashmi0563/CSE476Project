import Backend.config as config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = config.MODEL
token = config.HUGGINGFACE_TOKEN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

input_txt = "Explain gravity in simple terms:"
inputs = tokenizer(input_txt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0][len(input_txt):], skip_special_tokens=True))