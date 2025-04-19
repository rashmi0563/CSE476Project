import os
import torch
from datasets import load_dataset
from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import PeftModel
import evaluate
from tqdm import tqdm
import config
import gc
import LoRA_config as lora
from Backend.model import Load_model

test_adapter_path = config.DOLLY_SFT_OUTPUT_DIR
evaluation_dataset_path = config.EVALUATION_DATASET_PATH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer = Load_model(test_adapter_path)
tokenizer.padding_side = "left"

#load from local
eval_dataset = load_dataset("json", data_files=evaluation_dataset_path, split="train")

# load from huggingface 
#eval_dataset = load_dataset(evaluation_dataset_path, split="validation")


