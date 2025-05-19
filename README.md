# Uplift-RAG
The implementation for EMNLP 2025 Submission: Uplift-RAG: Uplift-Driven Knowledge Preference Alignment for Retrieval-Augmented Generation.

## How to run the code

1.Download the roberta-base from [Huggingface](https://huggingface.co/FacebookAI/roberta-base).

2.Run Training:
	> python train_reranker.py --pretrained_model_path="roberta-base" --learning_rate=1e-5 --dataset="dataset/web_questions" --gpu=0 --infer="False" --cls_loss --pair_loss --list_loss 

3.Run Inference:
    > python train_reranker.py --pretrained_model_path="roberta-base" --learning_rate=1e-5 --dataset="dataset/web_questions" --gpu=0 --infer="True" --cls_loss --pair_loss --list_loss --weight=""

You can specify the gpu id, learning_rate, used dataset,  and loss_types by cmd line arguments.

## Downstream Evaluation
After obtaining the reranker-selected and ordered documents through inference, we evaluate the results using FlashRAG, a widely used framework for Retrieval-Augmented Generation (RAG). You can find the FlashRAG code repository [here](https://github.com/RUC-NLPIR/FlashRAG).


## Environment

Our experimental environment is shown below:

```
numpy version: 1.26.4
pandas version: 2.2.3
torch version: 2.5.1
transformers version: 4.47.1
```
