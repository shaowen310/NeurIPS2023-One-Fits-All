seq_len=96
label_len=0
model=GPT4TS
gpt_layer=6

for percent in 100
do
# for pred_len in 24 36 48 96 192
for pred_len in 96
do

model_id=ETTh2_${model}_${gpt_layer}_${seq_len}_${pred_len}_$percent

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id $model_id \
    --data ett_h \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --batch_size 256 \
    --is_training 0 \
    --decay_fac 0.5 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 1 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer $gpt_layer \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 20 \
    --is_gpt 1 \
    2>&1 | tee -a logs/${model_id}_ETTh1.log

done
done
