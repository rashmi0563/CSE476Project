from datasets import load_dataset, concatenate_datasets
import config

class inst_DatasetLoader:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
    
    def load_alpaca(self):
        dataset_path = "tatsu-lab/alpaca"
        dataset = load_dataset(dataset_path, split="train")
        return dataset.map(self.format_alpaca)
    
    def load_dolly(self):
        dataset_path = "databricks/databricks-dolly-15k"
        dataset = load_dataset(dataset_path, split="train")
        return dataset.map(self.format_dolly)
    
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
        return concatenate_datasets([alpaca_dataset, dolly_dataset])
    
    def formatting_promopts_func(self, batch):
        return batch["formatted_text"]