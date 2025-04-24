HUGGINGFACE_TOKEN = ""
MODEL = "meta-llama/Llama-3.2-3B"

#Alpaca Dataset SFT(instruction tuning) 1st
Alpaca_SFT_OUTPUT_DIR = '../llama-3.2-3B-sft'
Alpaca_Dataset_path = 'tatsu-lab/alpaca'

#DOLLY Dataset SFT(instruction tuning) 2nd
DOLLY_SFT_OUTPUT_DIR = "../llama-3.2-3B-sft-dolly"
DOLLY_Dataset_path = 'databricks/databricks-dolly-15k'


EVALUATION_OUTPUT_DIR = "./evaluation_results"