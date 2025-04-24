import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging
)

from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import config
import LoRA_config as lora
from inst_DatasetLoader import inst_DatasetLoader

#=========model load==================
base_model_name = config.MODEL
base_sft_output_dir = "/home/jpark284/CSE476/adapter"
new_sft_output_dir = "sft_a_d1"
sft_output_dir = os.path.join(base_sft_output_dir, new_sft_output_dir)
os.makedirs(sft_output_dir, exist_ok=True)
# Huggingface token
hf_token = config.HUGGINGFACE_TOKEN
dataset_path = "tatsu-lab/alpaca"

# QLoRa Setting
bnb_config = lora.bnb_config
lora_config = lora.lora_config

#Llama3.2-3b model load
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config = bnb_config,
    device_map={"" : 0},
    token=hf_token,
    trust_remote_code=True
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    token=hf_token,
    trust_remote_code = True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#======= dataset load and formatting=========
# dataset = load_dataset(dataset_path, split="train")

# def formatting_prompts_func(batch):
#     if not isinstance(batch['instruction'], (list, tuple)):
#         batch = {k: [v] for k, v in batch.items()}
#     if 'text' in batch:
#         texts = batch['text']
#         if not isinstance(texts, (list, tuple)):
#             texts = [texts]
#         output_texts = [t + tokenizer.eos_token for t in texts]
#     else:
#         output_texts = []
#         for instruction, input_text, output_text in zip(batch['instruction'], batch['input'], batch['output']):
#             text = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
#             text += f"### Instruction:\n{instruction}\n\n"
#             if input_text:
#                 text += f"### Input:\n{input_text}\n\n"
#             text += f"### Response:\n{output_text}"
#             output_texts.append(text + tokenizer.eos_token)
#     return output_texts


dataset_loader = inst_DatasetLoader(tokenizer)
dataset = dataset_loader.load_combined_dataset()
formatting_prompts_func = dataset_loader.formatting_promopts_func




training_arguments = TrainingArguments(
    output_dir = sft_output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_arguments,
    #max_seq_length=1024,
)

print("=== Train Start ===")
trainer.train()
print("=== Train Complete ===")

trainer.save_model(sft_output_dir)
tokenizer.save_pretrained(sft_output_dir)
print(f"=== Save SFT-Model and Tokenizer Complete ===")

del model
del trainer
torch.cuda.empty_cache()
