import config as config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
class Load_model():
    def __init__(self,adapter_path):
        self.base_model_name = config.MODEL
        self.adapter_path = adapter_path
        self.token = config.HUGGINGFACE_TOKEN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            token=self.token,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, 
            token=self.token, 
            trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model = self.model.eval()

        print(f"--- SFT model load complete: {self.base_model_name} + {self.adapter_path} ---")
    
    def get(self):
        return self.model, self.tokenizer
