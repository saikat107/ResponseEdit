#!/bin/bash
src=$1
tgt=$2
model=$3
other_op=$4
echo $other_op
python edit/train.py \
	-save_path $model \
	-log_home log_tmp \
	-online_process_data \
	-train_src $src \
	-train_tgt $tgt \
	-layers 1 -enc_rnn_size 512 -brnn -word_vec_size 300 -dropout 0.5 \
	-batch_size 2 -beam_size 1 \
	-epochs 2000 \
	-gpus 0 \
	-optim adam -learning_rate 0.001 \
	-curriculum 0 -extra_shuffle \
	-seed 12345 -cuda_seed 12345 $other_op

