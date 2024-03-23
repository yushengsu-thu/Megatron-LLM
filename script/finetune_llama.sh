# downlaod data

DATASET="startcoder"
DATA_PREFIX="/lustre/scratch/shared-folders/llm_project/yusheng/mt/Megatron-LLM/sft_data"
VOCAB_PREFIX="/lustre/scratch/shared-folders/llm_project/yusheng/mt/Megatron-LLM/llm/llama/llama-2-7b/llama-2-7b_hf"

OUTPUT_DIR=$DATA_PREFIX/$DATASET/"tokenized"
mkdir $OUTPUT_DIR

'''
###################
## data preprocessing
#python3 ../tools/preprocess_instruct_data.py \
#	--input $DATA_PREFIX/$DATASET/data.jsonl \
#	--output_prefix $OUTPUT_DIR \
#	--tokenizer_type SentencePieceTokenizer \
#	--vocab_file $VOCAB_PREFIX/tokenizer.model \
#	--chunk_size 64 \
#	--workers 64 \
#	--vocab_extra_ids_list "<|im_start|>,<|im_end|>" \
#	--question_key question \
#	--answer_key response \
#	--system_key system_prompt  # Optional
###################


python ../tools/preprocess_data.py \
    --input $DATA_PREFIX/$DATASET/raw.jsonl \
	--output_prefix $OUTPUT_DIR/starcoder \
	--tokenizer_type SentencePieceTokenizer \
	--vocab_file $VOCAB_PREFIX/tokenizer.model \
	--chunk_size 32 \
	--workers 64 \
	--no_new_tokens


#Weight conversion
LLM_LOAD_DIR="/lustre/scratch/shared-folders/llm_project/yusheng/mt/Megatron-LLM/llm/llama/llama-2-7b/llama-2-7b_hf/"
LLM_SAVE_DIR="/lustre/scratch/shared-folders/llm_project/yusheng/mt/Megatron-LLM/llm/llama/llama-2-7b/llama-2-7b_mt/"

# Model Convert
python3 ../weights_conversion/hf_to_megatron.py \
    --model llama2 \
    --size 7 \
	--out $LLM_SAVE_DIR \
    --model-path $LLM_LOAD_DIR \
    --cache-dir $LLM_LOAD_DIR






# Correctness verification (optional)
# arguments required by `torchrun`
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"

LLM_LOAD_DIR="/lustre/scratch/shared-folders/llm_project/yusheng/mt/Megatron-LLM/llm/llama/llama-2-7b/llama-2-7b_mt/"
LLM_SAVE_DIR="/lustre/scratch/shared-folders/llm_project/yusheng/mt/Megatron-LLM/llm/llama/llama-2-7b/llama-2-7b_mt_sft/"
TENSORBOARD_DIR="/lustre/scratch/shared-folders/llm_project/yusheng/mt/Megatron-LLM/tensorboard/"
DATA_DIR=$OUTPUT_DIR/starcoder_text_document # without the .idx or .bin extension
VOCAB_PREFIX="/lustre/scratch/shared-folders/llm_project/yusheng/mt/Megatron-LLM/llm/llama/llama-2-7b/llama-2-7b_mt"

torchrun $DISTRIBUTED_ARGS ../verify_correctness.py \
	--model_name llama2 \
	--model_size 7 \
	--load $LLM_LOAD_DIR \
	--data_path $DATA_DIR \
	--tokenizer_type SentencePieceTokenizer \
	--vocab_file $VOCAB_PREFIX/tokenizer.model \
	--huggingface_cache $LLM_LOAD_DIR \
	--huggingface_device cuda:1 \
	$COMMON_ARGS $LLAMA_ARGS  # dont include LLAMA_ARGS if using Falcon

    # --data_path $DATA_DIR/starcoder_text_document \ --> # without the .idx or .bin extension


'''




#Model sharding

python3 ../tools/checkpoint_util.py \
	--target_tensor_parallel_size 2 \
	--target_pipeline_parallel_size 1 \
	--load_dir /path/to/megatron/weights/ \
	--save_dir /path/to/sharded/weights/ \
	--model_type llama2 \
	--true_vocab_size 32000 \
	--bf16


exit


NUMBER_OF_GPUS_for_EACH_NODE=4
#NUMBER_OF_GPUS_for_EACH_NODE=1
NUMBER_OF_NODES=1
NODE_ID=0
#localhost=172.23.30.9

LLM_LOAD_DIR="/lustre/scratch/shared-folders/llm_project/yusheng/mt/Megatron-LLM/llm/llama/llama-2-7b/llama-2-7b_mt/"
LLM_SAVE_DIR="/lustre/scratch/shared-folders/llm_project/yusheng/mt/Megatron-LLM/llm/llama/llama-2-7b/llama-2-7b_mt_sft/"
TENSORBOARD_DIR="/lustre/scratch/shared-folders/llm_project/yusheng/mt/Megatron-LLM/tensorboard/"
DATA_DIR=$DATA_DIR
VOCAB_PREFIX="/lustre/scratch/shared-folders/llm_project/yusheng/mt/Megatron-LLM/llm/llama/llama-2-7b/llama-2-7b_mt"


# training or tuning
LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 50"
TRAIN_ARGS="--train_iters 6500 --lr_decay_style cosine --lr_warmup_iters 650 --lr 2e-5 --min_lr 2e-6"
DISTRIBUTED_ARGS="--nproc_per_node $NUMBER_OF_GPUS_for_EACH_NODE --nnodes $NUMBER_OF_NODES --node_rank $NODE_ID --master_addr localhost --master_port 6000"
torchrun $DISTRIBUTED_ARGS ../finetune.py \
	--tensor_model_parallel_size 2 \
	--pipeline_model_parallel_size 1 \
	--load $LLM_LOAD_DIR \
	--save $LLM_SAVE_DIR \
	--tensorboard_dir $TENSORBOARD_DIR \
	--data_path $DATA_DIR \
	--model_name llama2 \
	--tokenizer_type SentencePieceTokenizer \
	--vocab_file $VOCAB_PREFIX/tokenizer.model \
	--bf16 \
	--use_flash_attn \
	--micro_batch_size 8 \
	--global_batch_size 64 \
	--sequence_parallel \
	--recompute_granularity selective \
	--use_checkpoint_args \
	--data_type instruction \
	--variable_seq_lengths \
	--vocab_extra_ids_list "<|im_start|>,<|im_end|>" \
	$COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $LLAMA_ARGS
