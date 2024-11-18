# export CUDA_VISIBLE_DEVICES=0

seq_len=96
label_len=0
model=GPT4TS
gpt_layer=6

for percent in 100
do
for pred_len in 24 36 48 96 192
do

model_id=weather_${model}_${gpt_layer}_${seq_len}_${pred_len}_$percent

python main.py \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id $model_id \
    --data custom \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --batch_size 512 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.9 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 21 \
    --c_out 21 \
    --freq 0 \
    --lradj type3 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer $gpt_layer \
    --itr 3 \
    --model $model \
    --is_gpt 1 \
    2>&1 | tee -a logs/$model_id.log
    
done
done
