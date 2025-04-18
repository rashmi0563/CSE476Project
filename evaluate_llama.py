import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from Backend.model import Load_model

# Load dev dataset
with open("dev_data.json", "r") as f:
    data = json.load(f)

# Load model
model_loader = Load_model()
model, tokenizer = model_loader.get()
model.to("cuda")
model.eval()

# Extract final answer after '####'
def extract_answer(text):
    match = re.search(r"####\s*([\d\.\-]+)", text)
    return match.group(1).strip() if match else None

correct = 0
total = len(data)

for sample in tqdm(data):
    prompt = f"You are a helpful assistant. Answer the following question.\n\nQuestion: {sample['question']}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_answer(model_output)
    actual = extract_answer(sample["answer"])

    if pred == actual:
        correct += 1

accuracy = correct / total * 100
print(f"Evaluation complete: {correct}/{total} correct")
print(f"Accuracy: {accuracy:.2f}%")
