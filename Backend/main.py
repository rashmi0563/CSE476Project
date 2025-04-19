from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import Load_model
import config
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
adapter = config.DOLLY_SFT_OUTPUT_DIR
model_loader = Load_model(adapter)
model, tokenizer = model_loader.get()


class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(req: PromptRequest):
    q = req.prompt
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    prompt += f"### Instruction:\n{q}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #input_ids = inputs["input_ids"]
    #attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id
            )
    #output_token_ids = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:")[-1].strip()
    return {"response" : generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)