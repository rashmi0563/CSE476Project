from fastapi import FastAPI
from pydantic import BaseModel
from model import Load_model
import torch

app = FastAPI()

model_loader = Load_model()
model, tokenizer = model_loader.get()


class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(req: PromptRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response" : generated_text}