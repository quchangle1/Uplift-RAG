import random
import os
import jsonlines
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
import os
import argparse
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)


class JoinedDataset(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.data[i] for i in range(*idx.indices(len(self)))]
        return self.data[idx]


class Collater:
    def __init__(self, tokenizer, is_train) -> None:
        self._tokenizer = tokenizer
        self.is_train = is_train

    def tokenize(self, *args):
        return self._tokenizer(
            *args,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def collate_training_data(self, batch_samples):
        res = {
            "classification": [
                self.tokenize(
                    [
                        [sample["question"], doc]
                        for sample in batch_samples
                        for doc in sample["retrieval_docs"]
                    ]
                ),
                torch.tensor(
                    [label for sample in batch_samples for label in sample["uplift_label"]]
                ),
            ],
            "lists":[
                self.tokenize(
                    [
                        [sample["question"], doc]
                        for sample in batch_samples
                        for doc in sample["retrieval_docs"]
                    ]
                ),
                torch.tensor(
                    [f1 for sample in batch_samples for f1 in sample["f1"]]
                ),
            ]
        }
        return res

    def collate_final_test_data(self, batch_samples):
        return(
                [sample['query_id'] for sample in batch_samples],
                [sample['question'] for sample in batch_samples],
                [sample['retrieval_docs'] for sample in batch_samples],
                self.tokenize(
                    [
                        [sample["question"], doc]
                        for sample in batch_samples
                        for doc in sample["retrieval_docs"]
                    ]
                )
        )

    def __call__(self, batch_samples):
        if self.is_train:
            return self.collate_training_data(batch_samples)
        else:
            return self.collate_final_test_data(batch_samples)