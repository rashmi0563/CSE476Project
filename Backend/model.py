import config as config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
class Load_model():
    def __init__(self):
        self.model_name = config.MODEL
        self.token = config.HUGGINGFACE_TOKEN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.token)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.token, torch_dtype= torch.float16).to(self.device)
    
    def get(self):
        return self.model, self.tokenizer
