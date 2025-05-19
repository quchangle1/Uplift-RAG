from enum import Enum, auto
import random
from typing import List
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
import os
import argparse
from sklearn.metrics import f1_score
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification



class JoinedModelLoss(Enum):
    ClaasificationLoss = auto()
    PairLoss = auto()
    ListLoss = auto()


class JoinedModel(nn.Module):
    def __init__(self, pretrained_model_path, loss_types: List[JoinedModelLoss]):
        super(JoinedModel, self).__init__()
        self.loss_types = loss_types
        self.roberta = AutoModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 2)
        self.weights_calculator = None

    def forward(self, cls_tokens):
        outputs = self.roberta(**cls_tokens).last_hidden_state[:, 0, :]
        outputs = self.classifier(outputs)
        return outputs