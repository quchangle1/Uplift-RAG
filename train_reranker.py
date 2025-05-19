import random
import os
from typing import Dict
import numpy as np
import json
import torch
from torch import nn
import torch.nn.functional as F
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

from joined_dataset import JoinedDataset, Collater
from joined_model import JoinedModel, JoinedModelLoss
from torch.amp import autocast, GradScaler
import logging


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        loss_types,
        optimizer,
        device,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.loss_types = loss_types
        self.optimizer = optimizer
        self.device = device

    def calc_cls_loss(self, cls_tokens, labels):
        outputs = self.model.roberta(**cls_tokens).last_hidden_state[:, 0, :]
        outputs = self.model.classifier(outputs)
        loss_cls = nn.functional.cross_entropy(outputs, labels, label_smoothing=0.1)
        return loss_cls

    def calc_pairwise_loss(self, rank_tokens, f1_scores):
        num_docs = 10
        batch_size = f1_scores.shape[0] // 10

        outputs = self.model.roberta(**rank_tokens).last_hidden_state[:, 0, :]
        outputs = self.model.classifier(outputs)[:, 1]

        loss_rank = 0
        pair_count = 0 

        for b in range(batch_size):
            f1_batch = f1_scores[b * num_docs: (b + 1) * num_docs] 
            output_batch = outputs[b * num_docs: (b + 1) * num_docs]  

            sorted_indices = torch.argsort(f1_batch, descending=True)

            for m in range(num_docs):
                for n in range(m + 1, num_docs):
                    i, j = sorted_indices[m], sorted_indices[n]  
                    if f1_batch[i] != f1_batch[j]:  
                        loss_rank += -nn.functional.logsigmoid(output_batch[i] - output_batch[j])
                        pair_count += 1
        return loss_rank / pair_count if pair_count > 0 else torch.tensor(0.0, device=outputs.device)


    def calc_list_loss(self, list_tokens, f1_scores):
        batch_size = f1_scores.shape[0] // 10 
        outputs = self.model.roberta(**list_tokens).last_hidden_state[:, 0, :] 
        scores = self.model.classifier(outputs)[:, 1]  
        target_probs = F.softmax(f1_scores.view(batch_size, 10).float(), dim=-1)
        pred_probs = F.softmax(scores.view(batch_size, 10).float(), dim=-1)
        loss = F.kl_div(pred_probs.log(), target_probs, reduction="batchmean")

        return loss

    def cal_loss(self, classification, lists):
        losses = []
        alpha_1 = 0.1
        alpha_2 = 0.1 
        if JoinedModelLoss.ClaasificationLoss in self.loss_types:
            X, labels = classification
            loss_cls = self.calc_cls_loss(X, labels)
            losses.append(alpha_1 * loss_cls)
        
        if JoinedModelLoss.PairLoss in self.loss_types:
            list_tokens, f1_scores = lists
            loss_pair = self.calc_pairwise_loss(list_tokens, f1_scores)
            losses.append(alpha_2 *loss_pair)

        if JoinedModelLoss.ListLoss in self.loss_types:
            list_tokens, f1_scores = lists
            loss_list = self.calc_list_loss(list_tokens, f1_scores)
            losses.append(loss_list)

        return sum(losses)

    def train_loop(self, epoch, total_loss):
        progress_bar = tqdm(range(len(self.train_dataloader)))
        progress_bar.set_description(f"loss: {0:>7f}")
        finish_step_num = (epoch - 1) * len(self.train_dataloader)

        self.model.train()
        scaler = GradScaler()

        for step, sample in enumerate(self.train_dataloader, start=1):
            for k, v in sample.items():
                if isinstance(v, list):
                    sample[k] = [vi.to(self.device) for vi in v]
                else:
                    sample[k] = v.to(self.device)

            self.optimizer.zero_grad()
            
            with autocast("cuda", dtype=torch.float16):
                loss = self.cal_loss(**sample)
            
            scaler.scale(loss).backward()
            scaler.step(self.optimizer) 
            scaler.update() 

            total_loss += loss.item()
            progress_bar.set_description(
                f"loss: {total_loss/(finish_step_num + step):>7f}"
            )
            progress_bar.update(1)
        return total_loss
        
    def test_loop(self, dataloader, outdir, dataset_type="Test"):
        assert dataset_type in ["Train", "Valid", "Test"]
        log_file = os.path.join(outdir, "test_log.txt")
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.model.eval()
        tp, fp, fn = 0, 0, 0
        y0 = y1 = 0
        with torch.no_grad():
            for sample in tqdm(dataloader):
                X, y = sample["classification"]
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X).argmax(1)
                tp += torch.sum((pred == 1) & (y == 1)).item()
                fn += torch.sum((pred == 0) & (y == 1)).item()
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError as e:
            print(e)
            print(f"n_y == 0: {y0}\nn_y == 1: {y1}")
            return 0

        logging.info(f"{dataset_type} dataset recall: {(100*recall):>0.1f}%")
        logging.info(f"n_y == 0: {y0}\nn_y == 1: {y1}")
        return recall


    def train(self, epoch_num, outdir):
        total_loss = 0.0
        best_f1 = 0.0
        os.makedirs(outdir, exist_ok=True)
        for t in range(epoch_num):
            log_file = os.path.join(outdir, "test_log.txt")
            logging.basicConfig(
                filename=log_file,
                filemode='a',
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            logging.info(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
            total_loss = self.train_loop(t + 1, total_loss)
            valid_f1 = self.test_loop(self.valid_dataloader, outdir, dataset_type="Valid")
            if valid_f1 > best_f1:
                best_f1 = valid_f1
            print("saving new weights...\n")
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    outdir,
                    f"epoch_{t+1}_valid_f1_{(100*valid_f1):0.1f}_model_weights.bin",
                ),
            )
        self.test_loop(self.test_dataloader, outdir, dataset_type="Test")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--cls_loss", action="store_true")
    parser.add_argument("--pair_loss", action="store_true")
    parser.add_argument("--list_loss", action="store_true")
    parser.add_argument("--infer", default="False", type=str, help="train or infer")
    parser.add_argument("--weights", type=str, help="Path to model weights for inference")
    args = parser.parse_args()
    print(f"Using device: gpu:{args.gpu}")

    return args


def seed_everything(seed=2025):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = get_args()
    seed_everything(42)
    learning_rate = args.learning_rate
    batch_size = 4
    epoch_num = 10
    device = f"cuda:{args.gpu}"
    model_name = os.path.basename(args.pretrained_model_path)
    dataset = os.path.basename(args.dataset)
    args_losses = "_".join([loss for loss in ["cls" if args.cls_loss else "", "pair" if args.pair_loss else "", "list" if args.list_loss else ""] if loss])
    outdir = f"{dataset}_{model_name}_{args_losses}_lr{learning_rate}_bs{batch_size}"
    train_data_path = os.path.join(args.dataset, "train.json")
    valid_data_path = os.path.join(args.dataset, "dev.json")
    test_data_path = os.path.join(args.dataset, "test.json")
    if args.infer == 'False':
        checkpoint = args.pretrained_model_path
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=512)

        train_data = JoinedDataset(train_data_path)
        collater = Collater(tokenizer, is_train=True)
        train_dataloader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, collate_fn=collater
        )
        valid_data = JoinedDataset(valid_data_path)
        valid_dataloader = DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, collate_fn=collater
        )
        test_data = JoinedDataset(test_data_path)
        test_dataloader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, collate_fn=collater
        )
        loss_types = []
        if args.cls_loss:
            loss_types.append(JoinedModelLoss.ClaasificationLoss)
        if args.list_loss:
            loss_types.append(JoinedModelLoss.ListLoss)
        if args.pair_loss:
            loss_types.append(JoinedModelLoss.PairLoss)
        print(loss_types)
        model = JoinedModel(checkpoint, loss_types)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            loss_types=loss_types,
            optimizer=optimizer,
            device=device,
        )

        trainer.train(epoch_num=epoch_num, outdir=outdir)
        print("Done!")
    else:
        model = JoinedModel(args.pretrained_model_path, loss_types=[])
        model.load_state_dict(torch.load(args.weights))
        model.eval()
        model.to(device)
        test_data = JoinedDataset(test_data_path)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, model_max_length=512)
        collater = Collater(tokenizer, is_train=False)
        test_dataloader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, collate_fn=collater
        )
        results = []
        with torch.no_grad():
            for sample in tqdm(test_dataloader):
                query_ids, questions, docs, tokenized_query_doc = sample
                tokenized_query_doc = tokenized_query_doc.to(device)
                outputs = model(tokenized_query_doc)
                scores = torch.softmax(outputs, dim=1)[:, 1]
                index = 0
                processed_batch = []
                batch_size = len(query_ids)
                for i in range(batch_size):
                    query_id = query_ids[i]
                    question = questions[i]
                    retrieval_docs = docs[i]
                    doc_scores = scores[index : index + len(retrieval_docs)]
                    index += len(retrieval_docs)
                    filtered_docs = [
                        (doc, score.item()) for doc, score in zip(retrieval_docs, doc_scores) if score >0.5
                    ]
                    filtered_docs.sort(key=lambda x: x[1], reverse=True)
                    processed_batch.append({
                        "query_id": query_id,
                        "question": question,
                        "retrieval_result": [doc for doc, _ in filtered_docs]
                    })
                results.extend(processed_batch)    
        converted = []
        for item in results:
            new_item = {
                "id": item["query_id"],
                "question": item["question"],
                "retrieval_result": [{"contents": doc} for doc in item["retrieval_result"]]
            }
            converted.append(new_item)
        with open("test_results.json", "w") as f:
            json.dump(converted, f, indent=4)

        

if __name__ == "__main__":
    main()
