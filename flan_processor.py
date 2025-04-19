import torch
from datasets import load_dataset, concatenate_datasets
import random
from tqdm import tqdm
import config

class FlanDatasetProcessor:
    def __init__(self,
                 dataset_path=config.FLAN_Dataset_path,
                 categories=config.flan_categories,
                 samples_per_category=1000,
                 tokenizer=None,
                 hf_token=None,
                 cache_dir=None,
                 input_col='inputs',
                 output_col='targets'
                 ):
        self.dataset_path = dataset_path
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
        formatted_texts = [self._format_prompt(instr, resp) for instr, resp in zip(instructions, responses)]
        return {"text" : formatted_texts}

    def load_and_process(self):
        print("--- FLAN Dataset Processing start (FlanDatasetProcessor) ---")
        sampled_datasets = []
        print(f"--- Total {len(self.categories)} categories processing ---")
        for category in tqdm(self.categories, desc="FLAN 카테고리 처리 중"):
            try:
                subset_dataset = load_dataset(
                    self.dataset_path,
                    category,
                    split='train',
                    use_auth_token=self.hf_token,
                    cache_dir=self.cache_dir
                )
                if self.input_col not in subset_dataset.column_names or \
                   self.output_col not in subset_dataset.column_names:
                    print(f"Warn: '{category}' do not have '{self.input_col}' or '{self.output_col}' column, skip!")
                    continue

                actual_size = len(subset_dataset)
                sample_size = min(self.samples_per_category, actual_size)

                if sample_size > 0:
                    sampled_subset = subset_dataset.shuffle(seed=self.seed).select(range(sample_size))
                    sampled_datasets.append(sampled_subset)
                else:
                    pass

            except Exception as e:
                print(f"=====Error {category} : {e} =======")
        if not sampled_datasets:
            raise RuntimeError("No FLAN Dataset to process, check category name or dataset")
        
        print(f"=== total {len(sampled_datasets)} category sampling complete =======")
        final_raw_dataset = concatenate_datasets(sampled_datasets)
        print(f"--- Dataset Concatenation Complete : Total {len(final_raw_dataset)} Sample ---")

        print("=== Data Formatting===")
        processed_dataset = final_raw_dataset.map(
            self._process_batch,
            batched=True,
            remove_columns=final_raw_dataset.column_names
        )

        print(f"=== Final FLAN Training Dataset Generation Complete(via processor)")
        print(f" Total {len(processed_dataset)} sample")
        print(f"Dataset features: {processed_dataset.features}")

        return processed_dataset
