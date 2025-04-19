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

#=========model load==================
first_train = False #set to True if you train model for first time(means you got no adapter)
base_model_name = config.MODEL

prev_adapter_path = config.Alpaca_SFT_OUTPUT_DIR
next_adapter_path = config.DOLLY_SFT_OUTPUT_DIR
dataset_path = config.DOLLY_Dataset_path

hf_token = config.HUGGINGFACE_TOKEN

bnb_config = lora.bnb_config # QLoRa Setting
lora_config = lora.lora_config #LoRA Setting

#Llama3.2-3b model load
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
dataset = load_dataset(dataset_path, split="train")

def formatting_prompts_func_dolly(batch):

    instructions = batch['instruction']
    contexts = batch['context']
    responses = batch['response']
    if not isinstance(instructions, (list, tuple)):
        batch = {k: [v] for k, v in batch.items()}
    output_texts = []
    prompt_template_start = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"

    for instruction, context, response in zip(instructions, contexts, responses):
        text = prompt_template_start
        text += f"### Instruction:\n{instruction}\n\n"
        if context:
            text += f"### Input:\n{context}\n\n"
        text += f"### Response:\n{response}"
        output_texts.append(text + tokenizer.eos_token)
    return output_texts


# ======== LoRA setting ============


training_arguments = TrainingArguments(
    output_dir = next_adapter_path,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=10,
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
    train_dataset=dataset,
    formatting_func=formatting_prompts_func_dolly,
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
print(f"=== Save DOlly SFT-Model and Tokenizer Complete ===")

del model
del trainer
torch.cuda.empty_cache()
