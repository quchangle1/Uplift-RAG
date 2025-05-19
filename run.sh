python train_reranker.py \
--pretrained_model_path="PLM/roberta-base" \
--learning_rate=1e-5 \
--dataset="dataset/web_questions" \
--gpu=5 \
--infer="False" \
--cls_loss \
--pair_loss \
--list_loss 
