import torch
from datasets import load_dataset, concatenate_datasets
import random
from tqdm import tqdm
import config

class FlanDatasetProcessor:
    def __init__(self,
                 dataset_repo_id=config.FLAN_Dataset_path,
                 categories=config.flan_categories,
                 samples_per_category=1000,
                 tokenizer=None,
                 hf_token=None,
                 cache_dir=None,
                 input_col='inputs',
                 output_col='targets',
                 task_col='task'
                 ):
        self.dataset_repo_id = dataset_repo_id
        self.categories = categories
        self.samples_per_category = samples_per_category
        self.tokenizer = tokenizer
        self.hf_token = hf_token
        self.input_col = input_col
        self.output_col = output_col
        self.cache_dir = cache_dir

    def _format_prompt(self, instruction, response):
        text = f"### Instruction:\n{instruction}\n\n"
        text += f"### Response:\n{response}"
        return text + self.tokenizer.eos_token
    
    def _process_batch(self, batch):
        instructions = batch[self.input_col]
        responses = batch[self.output_col]
        formatted_texts = [self._format_prompt(instr, resp) 
                           for instr, resp in zip(instructions, responses)]
        return {"text" : formatted_texts}

    def load_and_process(self):
        print("--- FLAN Dataset Processing start (FlanDatasetProcessor) ---")
        sampled_datasets = []
        print(f"--- Processing {len(self.categories)} target tasks by loading individual files ---")
        base_data_path_pattern = f"hf://datasets/{self.dataset_repo_id}/train/{{category}}_train.jsonl"

        for category in tqdm(self.categories, desc="Loading & Sampling Tasks"):
            try:
                data_file_path = base_data_path_pattern.format(category=category)
                subset_dataset = load_dataset(
                    "json", # 타입을 json으로 명시
                    data_files=data_file_path, # 개별 파일 경로 지정
                    split='train', # 단일 파일을 로드하므로 split='train' 고정
                    cache_dir=self.cache_dir,
                    # token=self.hf_token # 공개 json 파일 로드에는 보통 불필요
                )
                if self.input_col not in subset_dataset.column_names or \
                   self.output_col not in subset_dataset.column_names:
                    print(f"Warning: File for '{category}' at '{data_file_path}' does not contain required columns '{self.input_col}' or '{self.output_col}'. Found: {subset_dataset.column_names}. Skipping.")
                    continue
                actual_size = len(subset_dataset)
                if actual_size == 0:
                    print(f"Warning: No data found in file for task '{category}'. Skipping.")
                    continue
                sample_size = min(self.samples_per_category, actual_size)
                sampled_subset = subset_dataset.shuffle().select(range(sample_size))
                sampled_datasets.append(sampled_subset)
            except FileNotFoundError:
               print(f"Error: Could not find file at '{data_file_path}'. Check category name and dataset structure. Skipping.")
            except Exception as e:
                print(f"Error processing task '{category}' from file '{data_file_path}': {e}. Skipping.")
        
        if not sampled_datasets:
            raise RuntimeError("No data could be sampled. Check task names (categories) or dataset file paths.")
        
        print(f"--- Successfully loaded and sampled data from {len(sampled_datasets)} tasks ---")
        final_raw_dataset = concatenate_datasets(sampled_datasets)
        print(f"--- Dataset Concatenation Complete: Total {len(final_raw_dataset)} samples ---")
        print("=== Applying Data Formatting ===")
        columns_to_remove = list(final_raw_dataset.column_names)
        processed_dataset = final_raw_dataset.map(
            self._process_batch,
            batched=True,
            remove_columns=columns_to_remove
        )
        processed_dataset = processed_dataset.shuffle()
        print(f" Total {len(processed_dataset)} samples")
        print(f" Dataset features: {processed_dataset.features}")
        return processed_dataset