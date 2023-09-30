dataset="Elec2"
gpu_id='0'

# hyperparameters of Data Simulator
encoder_dim=64
encoder_l=2
decoder_dim=64
decoder_l=3
threshold=0.1
goggle_beta=2.0
goggle_lr=0.02
goggle_batch_size=64

tune_data_model="MLP"   #"LightGBM"   #"FTTransformer"

pred_model="MLP"   #"LightGBM"   #"FTTransformer"

# hyperparameters of MLP
MLP_lr=1e-4

# hyperparameters of LightGBM
is_linear_tree=Ture
LG_lr=0.05
LG_ff=0.9
LG_bfrac=1.0
LG_bfreq=30
LG_l1=0.5
LG_l2=0.9
LG_lambda=0.1
LG_leaves=31

# hyperparameters of FTTransformer
T_lr=8e-5
T_i_dim=128
T_n_head=8
T_att_blocks=8
T_attn_dropout=0.1
T_batch=128

python test_coda_data.py --dataset $dataset \
--goggle_batch_size $goggle_batch_size --goggle_lr $goggle_lr \
--encoder_l $encoder_l  --decoder_l $decoder_l \
--encoder_dim $encoder_dim --decoder_dim $decoder_dim \
--goggle_beta $goggle_beta --gpu_id $gpu_id \
--pred_model $pred_model \
--tune_data_model $tune_data_model \
--MLP_lr $MLP_lr \
--is_linear_tree $is_linear_tree \
--LG_lr $LG_lr --LG_ff $LG_ff --LG_bfrac $LG_bfrac --LG_bfreq $LG_bfreq \
--LG_l1 $LG_l1 --LG_l2 $LG_l2 --LG_lambda $LG_lambda --LG_leaves $LG_leaves \
--T_lr $T_lr --T_i_dim $T_i_dim --T_n_head $T_n_head \
--T_att_blocks $T_att_blocks --T_attn_dropout $T_attn_dropout --T_batch $T_batch