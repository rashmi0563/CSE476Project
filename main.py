import config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = config.MODEL
token = config.HUGGINGFACE_TOKEN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

input_txt = "Are Gin and tonic and Paloma both cocktails based on tequila?"
inputs = tokenizer(input_txt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
#outputs = model.generate(**inputs)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(input_txt)
print(decoded[len(input_txt):].strip())
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))