from datasets import load_dataset, concatenate_datasets
import config

class general_DatasetLoader:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
    
    def load_alpaca(self):
        dataset_path = "yahma/alpaca-cleaned"
        dataset = load_dataset(dataset_path, split="train")
        return dataset.map(self.format_alpaca)
    
    def load_dolly(self):
        dataset_path = "databricks/databricks-dolly-15k"
        dataset = load_dataset(dataset_path, split="train")
        return dataset.map(self.format_dolly)
    
    def load_custom_dataset(self):
        from datasets import Dataset
        data = [
    {
        "instruction": "Who are you?",
        "input": "",
        "output": "I'm a Large Language Model developed by the GG team from CSE476 @ASU."
    },
    {
        "instruction": "What's your name?",
        "input": "",
        "output": "I'm GG-LLM, created by the CSE476 team at ASU."
    },
    {
        "instruction": "Can you tell me who you are?",
        "input": "",
        "output": "Of course! I'm a large language model designed by the GG team in CSE476 @ASU."
    },
    {
        "instruction": "Tell me about yourself.",
        "input": "",
        "output": "Sure! I'm an AI assistant developed by students in CSE476, Arizona State University."
    },
    {
        "instruction": "Please introduce yourself.",
        "input": "",
        "output": "Hi there! I'm a language model trained by the GG team from ASU's CSE476 course."
    },
    {
        "instruction": "Who created you?",
        "input": "",
        "output": "I was developed by the GG team as part of the CSE476 course at Arizona State University."
    },
    {
        "instruction": "What's your background?",
        "input": "",
        "output": "I'm an AI language model developed by the GG team in the CSE476 class at ASU."
    },
    {
        "instruction": "What are you?",
        "input": "",
        "output": "I'm a custom-trained language model built by the GG team - CSE476 @ASU."
    },
]
        dataset = Dataset.from_list(data)
        return dataset.map(self.format_alpaca)

    
    def format_alpaca(self, example):
        """ Alpaca dataset formatting """
        text = f"### Instruction:\n{example['instruction']}\n\n"
        if example['input']:
            text += f"### Input:\n{example['input']}\n\n"
        text += f"### Response:\n{example['output']}"
        return {"formatted_text": text + self.tokenizer.eos_token}
    
    def format_dolly(self, example):
        """ Dolly dataset formatting """
        text = f"### Instruction:\n{example['instruction']}\n\n"
        text += f"### Response:\n{example['response']}"
        return {"formatted_text": text + self.tokenizer.eos_token}
    
    def load_combined_dataset(self):
        """ Load combined datasets """
        alpaca_dataset = self.load_alpaca()
        dolly_dataset = self.load_dolly()
        custom_dataset = self.load_custom_dataset()
        combined = concatenate_datasets([alpaca_dataset, dolly_dataset, custom_dataset])
        return combined.shuffle()
    
    def formatting_promopts_func(self, batch):
        return batch["formatted_text"]