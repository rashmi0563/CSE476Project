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
from flan_processor import FlanDatasetProcessor

#=========model load==================
base_model_name = config.MODEL
prev_adapter_path = config.A_D_SFT_DIR

base_sft_output_dir = "/home/jpark284/CSE476/adapter"
new_sft_output_dir = "sft_general"
next_adapter_path = os.path.join(base_sft_output_dir, new_sft_output_dir)
os.makedirs(next_adapter_path, exist_ok=True)

hf_token = config.HUGGINGFACE_TOKEN

bnb_config = lora.bnb_config # QLoRa Setting
lora_config = lora.lora_config #LoRA Setting

flan_categories = config.flan_categories
samples_per_category = 1000 #for flan

#Llama3.2-3b model load
print(f"====== Base Model Loading : {base_model_name} ============")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config = bnb_config,
    device_map={"" : 0},
    token=hf_token,
    trust_remote_code=True
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

model = PeftModel.from_pretrained(base_model, prev_adapter_path, is_trainable=True)
print(f"==== previous adapter load complete === : {prev_adapter_path}")

tokenizer = AutoTokenizer.from_pretrained(
    prev_adapter_path,
    token=hf_token,
    trust_remote_code = True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#======= dataset load and formatting=========
#dataset = load_dataset(dataset_path, split="train")
flan_processor = FlanDatasetProcessor(
    tokenizer=tokenizer,
    hf_token=hf_token,
)
final_dataset = flan_processor.load_and_process()

# ======== LoRA setting ============


training_arguments = TrainingArguments(
    output_dir = next_adapter_path,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_steps=1000,
    logging_steps=50,
    learning_rate=1e-4,
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

print("=== Loaded PEFT model's trainable parameter ===")
model.print_trainable_parameters()

trainer = SFTTrainer(
    model=model,
    train_dataset=final_dataset,
    peft_config=None if first_train else lora_config,
    processing_class=tokenizer,
    args=training_arguments,
    #max_seq_length=1024,
)

print("=== Dolly Dataset Train Start ===")
trainer.train()
print("=== Additional Train Complete ===")

trainer.save_model(next_adapter_path)
tokenizer.save_pretrained(next_adapter_path)
print(f"=== Save FLAN SFT-Model and Tokenizer Complete ===")

del model
del trainer
torch.cuda.empty_cache()
