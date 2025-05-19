python train_reranker.py \
--pretrained_model_path="PLM/roberta-base" \
--learning_rate=1e-5 \
--dataset="dataset/web_questions" \
--gpu=5 \
--infer="False" \
--cls_loss \
--pair_loss \
--list_loss 


# python train_bge_joined.py \
# --pretrained_model_path="/new_disk1/changle_qu/PLM/roberta-large" \
# --learning_rate=1e-5 \
# --train_data_path="/new_disk1/changle_qu/DPA-RAG-main/mydata/llama-3.2-1B/web_questions/web_questions_train_data.json" \
# --valid_data_path="/new_disk1/changle_qu/DPA-RAG-main/mydata/llama-3.2-1B/web_questions/web_questions_test_data.json" \
# --test_data_path="/new_disk1/changle_qu/DPA-RAG-main/mydata/llama-3.2-1B/web_questions/web_questions_test_data.json" \
# --gpu=5 \
# --outdir="/new_disk1/changle_qu/DPA-RAG-main/model/bge-v2/point_pair_list_base" \
# --tensorboard_log_dir="/new_disk1/changle_qu/DPA-RAG-main/log" \
# --infer="True" \
# --cls_loss \
# --pair_loss \
# --list_loss 

# python train_reranker.py \
# --pretrained_model_path="/new_disk1/changle_qu/PLM/roberta-base" \
# --learning_rate=1e-5 \
# --dataset="/new_disk1/changle_qu/DPA-RAG-main/mydata/qwen-2-7B/nq" \
# --gpu=5 \
# --infer="True" \
# --cls_loss \
# --pair_loss \
# --list_loss \
# --weights="/new_disk1/changle_qu/DPA-RAG-main/qwen-2-7B/qwen2_nq_train_data.json_roberta-base_cls_pair_list_lr1e-05_bs4/epoch_7_valid_f1_45.1_model_weights.bin" 