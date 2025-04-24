from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load
import numpy as np
import json
from Backend import config
from Backend.model import Load_model
import torch

# === Step 1: Evaluation questions ===
eval_questions = [
    "What is the capital of Japan?",
    "Who discovered gravity?",
    "What causes rainbows to appear?",
    "Tell me three interesting facts about the moon.",
    "How many states are in the United States?",
    "Write a short story about a time-traveling cat.",
    "Compose a poem about the ocean in four lines.",
    "Write a product review for a fictional smart fridge.",
    "Summarize the plot of Cinderella in three sentences.",
    "Rewrite this sentence to sound more formal: \u201cI messed up.\u201d",
    "How do I reset my Wi-Fi router?",
    "Give me step-by-step instructions to bake a chocolate cake.",
    "What should I bring on a hiking trip?",
    "How do I write a resume for my first job?",
    "What are some tips for studying more effectively?",
    "Write a Python function that reverses a string.",
    "Explain what an API is in simple terms.",
    "How do you center a div using CSS?",
    "What\u2019s the difference between a list and a tuple in Python?",
    "Give me an example of recursion.",
    "Who are you?",
    "What can you do?",
    "How were you trained?",
    "Where are you from?",
    "Why should people trust you?"
]

# === Step 2: Create Dataset ===
eval_dataset = Dataset.from_dict({"instruction": eval_questions})

# === Step 3: Generation Function ===
def generate_answers(model, tokenizer, questions, max_tokens=256):
    #generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    outputs = []
    for q in questions:
        prompt = f"### Instruction:\n{q}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_length=max_tokens, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.split("### Response:")[-1].strip()
        outputs.append(response)
    return outputs

# === Step 4: Evaluation Metrics ===
bleu = load("bleu")
rouge = load("rouge")
exact_match = lambda p, r: int(p.strip().lower() == r.strip().lower())

def evaluate_outputs(preds, refs):
    results = {}
    results["BLEU"] = bleu.compute(predictions=preds, references=refs)["bleu"]
    results["ROUGE-L"] = rouge.compute(predictions=preds, references=refs)["rougeL"]
    results["Exact Match"] = np.mean([exact_match(p, r) for p, r in zip(preds, refs)])
    return results

# === Step 6: Save Comparison to JSON ===
def save_comparison_to_json(questions, model1_outputs, model2_outputs, filename="model_comparison.json"):
    result_list = []
    for i, q in enumerate(questions):
        result_list.append({
            f"Question {i+1}": q,
            "Model 1": model1_outputs[i],
            "Model 2": model2_outputs[i]
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result_list, f, indent=4, ensure_ascii=False)

# === Step 5: Run Example ===
if __name__ == "__main__":
    
    model_loader1 = Load_model(config.DOLLY_SFT_DIR)
    model_loader2 = Load_model(config.A_D_SFT_DIR)
    model1, tokenizer1 = model_loader1.get()
    model2, tokenizer2 = model_loader2.get()
    

    outputs_1 = generate_answers(model1, tokenizer1, eval_questions)
    outputs_2 = generate_answers(model2, tokenizer2, eval_questions)

    results = evaluate_outputs(outputs_2, outputs_1)
    print("=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # Save to JSON
    save_comparison_to_json(eval_questions, outputs_1, outputs_2)
