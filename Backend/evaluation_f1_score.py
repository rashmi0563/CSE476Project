from pydantic import BaseModel
from model import Load_model
import torch
import math
from transformers import pipeline
import json

# loading model
model_loader = Load_model()
model, tokenizer = model_loader.get()
model.eval()

from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re
    import string
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    gold_tokens = normalize_answer(a_gold).split()
    pred_tokens = normalize_answer(a_pred).split()
    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())

    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return int(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

from transformers import TextGenerationPipeline
generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=model.device.index if torch.cuda.is_available() else -1)

# JSON 로드
with open("dev_data.json", "r") as f:
    data = json.load(f)

# F1 계산 함수 (위 normalize_answer, compute_f1 포함 필요)
total_f1 = 0.0
for i, item in enumerate(data[0:99]):
    question = item["question"]
    reference = item["answer"]

    # 모델로부터 예측 생성
    output = generator(question, max_new_tokens=100, do_sample=False)
    prediction = output[0]["generated_text"].replace(question, "").strip()
    f1 = compute_f1(reference, prediction)
    total_f1 += f1

    print(f"Qestion Number{i}: F1 Score = {f1:.3f}")
    print(f"Q: {question}")
    print(f"Predcition: {prediction}")
    print(f"Reference : {reference}")
    print()

average_f1 = total_f1 / 100
print(f"\nAverage F1 Score (0~99): {average_f1:.3f}")

