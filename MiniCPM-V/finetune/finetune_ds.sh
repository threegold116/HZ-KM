#!/bin/bash

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/home/jiangshixin/pretrained_model/MiniCPM-Llama3-V-2_5" # or openbmb/MiniCPM-V-2
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/home/jiangshixin/myproject/HZ-KM/train_data/jsonfiles/vrs_train_qa_cap.json"
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm
echo "HELLO"
out_model_name="test_vrsbench_tune_llm_only"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
export WANDB_MODE=offline
# deepspeed --include "localhost:0,1,2,3,4,5,6,7"   --master_port 29514 finetune.py  \
# torchrun $DISTRIBUTED_ARGS finetune.py  \
echo "HELLO2"
torchrun $DISTRIBUTED_ARGS  finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --num_train_epochs 3\
    --prediction_loss_only false \
    --bf16 false \
    --fp16 true \
    --do_train \
    --tune_vision false \
    --tune_llm true \
    --model_max_length 512 \
    --max_slice_nums 9 \
    --output_dir /home/jiangshixin/model/minicpmv/vrsbench/${out_model_name} \
    --logging_dir /home/jiangshixin/myproject/HZ-KM/logs/output_minicpmv2/${out_model_name} \
    --logging_strategy "steps" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --report_to "wandb" 
