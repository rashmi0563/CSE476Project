from pydantic import BaseModel
from model import Load_model
import torch
import math
import json


model_loader = Load_model()
model, tokenizer = model_loader.get()
model.eval()


with open("dev_data.json", "r") as f:
    data = json.load(f)


# Calculate perplexity
def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())


perplexities = []
for i, item in enumerate(data[0:100]):
    text = item["answer"]
    ppl = calculate_perplexity(text)
    perplexities.append(ppl)
    print(f"{i}: Perplexity = {ppl:.2f}")


average_ppl = sum(perplexities) / len(perplexities)
print(f"\nAverage Perplexity (0~99): {average_ppl:.2f}")

