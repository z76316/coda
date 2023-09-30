dataset="Elec2"   #"ONP"  #"Elec2"
gpu_id='0'
batch_size=64
dropout_rate=0.5

num_rnn_layer=16
latent_dim=8
hidden_dim=16
learning_rate=1e-4
bce_w=0
epoches=50

python train_correlation_predictor.py --dataset $dataset --batch_size $batch_size \
                            --num_rnn_layer $num_rnn_layer --latent_dim $latent_dim \
                            --hidden_dim $hidden_dim --learning_rate $learning_rate \
                            --dropout_rate $dropout_rate --epoches $epoches --bce_w $bce_w \
                            --gpu_id $gpu_id