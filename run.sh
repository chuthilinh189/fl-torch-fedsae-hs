# 1. WSN_DS:
nohup python -u initialize_env.py -data wsn_ds  > logger/wsn_ds_init.out 2>&1 &

# Thí nghiệm limited attack
nohup python main.py -d wsn_ds -m AE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m DAE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m SAE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m FedMSE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 2000 -agg FedMSE -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m SupAE -tbs 128 -vbs 1 -di 17 -lr 0.01 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 10 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m DualLossAE -tbs 128 -vbs 1 -di 17 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 1 -mc 2 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m DualLossAE -tbs 128 -vbs 1 -di 17 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 1 -mc 5 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m DualLossAE -tbs 128 -vbs 1 -di 17 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 1 -mc 10 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m DualLossAE -tbs 128 -vbs 1 -di 17 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 1 -mc 20 > logger/wsn_ds_log.out 2>&1 &

nohup python evaluate.py -d wsn_ds -m AE -tbs 128 -vbs 128 -di 17 -ep 2000 -tm 1.7 --model_dir saved_models/AE/average/wsn_ds --log_csv logs/AE/wsn_ds_AE_ > logger/wsn_ds_AE_2000_1.7.log 2>&1
nohup python evaluate.py -d wsn_ds -m DAE -tbs 128 -vbs 128 -di 17 -ep 2000 -tm 3.0 --model_dir saved_models/DAE/average/wsn_ds --log_csv logs/DAE/wsn_ds_DAE_ > logger/wsn_ds_DAE_2000_3.log 2>&1
nohup python evaluate.py -d wsn_ds -m SAE -tbs 128 -vbs 128 -di 17 -ep 2000 -tm 3 --model_dir saved_models/SAE/average/wsn_ds --log_csv logs/SAE/wsn_ds_SAE_ > logger/wsn_ds_SAE_2000_3.log 2>&1
nohup python evaluate.py -d wsn_ds -m FedMSE -tbs 128 -vbs 128 -di 17 -ep 2000 -tm 3 --model_dir saved_models/FedMSE/FedMSE/wsn_ds --log_csv logs/FedMSE/wsn_ds_FedMSE_ > logger/wsn_ds_FedMSE_2000_3.log 2>&1
nohup python evaluate.py -d wsn_ds -m DualLossAE -tbs 128 -vbs 1 -di 17 -ep 2000 -tm 0.4 --model_dir saved_models/DualLossAE2old/average/wsn_ds --log_csv logs/DualLossAE2old/wsn_ds_DualLossAE > logger/wsn_ds_DualLossAE2old_2000_0.4.log 2>&1

# Thí nghiệm của bài hội thảo FedSAE-HS
nohup python main.py -d wsn_ds -m AE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log1.out 2>&1 &
nohup python main.py -d wsn_ds -m DAE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log2.out 2>&1 &
nohup python main.py -d wsn_ds -m SAE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log3.out 2>&1 &
nohup python main.py -d wsn_ds -m SAE1 -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log4.out 2>&1 &
nohup python main.py -d wsn_ds -m FedMSE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500 -agg FedMSE -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log5.out 2>&1 &

nohup python evaluate.py -d wsn_ds -m AE -tbs 128 -vbs 128 -di 17 -ep 500 -tm 2 --model_dir saved_models/conf_5/AE/average/wsn_ds --log_csv logs/conf_5/wsn_ds_AE_ > logger/conf_5/wsn_ds_AE_2.log 2>&1 &
nohup python evaluate.py -d wsn_ds -m DAE -tbs 128 -vbs 128 -di 17 -ep 500 -tm 5 --model_dir saved_models/conf_5/DAE/average/wsn_ds --log_csv logs/conf_5/wsn_ds_DAE_ > logger/conf_5/wsn_ds_DAE_5.log 2>&1 &
nohup python evaluate.py -d wsn_ds -m SAE -tbs 128 -vbs 128 -di 17 -ep 500 -tm 5 --model_dir saved_models/conf_5/SAE/average/wsn_ds --log_csv logs/conf_5/wsn_ds_SAE_ > logger/conf_5/wsn_ds_SAE_5.log 2>&1 &
nohup python evaluate.py -d wsn_ds -m FedMSE -tbs 128 -vbs 128 -di 17 -ep 500 -tm 5 --model_dir saved_models/conf_5/FedMSE/FedMSE/wsn_ds --log_csv logs/conf_5/wsn_ds_FedMSE_ > logger/conf_5/wsn_ds_FedMSE_5.log 2>&1 &
nohup python evaluate.py -d wsn_ds -m SAE1 -tbs 128 -vbs 128 -di 17 -ep 500 -tm 5 --model_dir saved_models/conf_5/SAE1/average/wsn_ds --log_csv logs/conf_5/wsn_ds_SAE1_ > logger/conf_5/wsn_ds_SAE1_5.log 2>&1 &

# Thí nghiệm với tấn công label flipping và gaussian noise
nohup python main.py -d wsn_ds -m SDAE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m SDAE1 -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m AE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m DAE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m SAE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m SAE1 -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500`-agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m SDAE -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log.out 2>&1 &
nohup python main.py -d wsn_ds -m SDAE1 -tbs 128 -vbs 128 -di 17 -lr 0.01 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/wsn_ds_log.out 2>&1 &

# Thí nghiệm unseen attack
nohup python main.py -d wsn_ds -m PTLAE -tbs 128 -vbs 1 -di 17 -lr 0.001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log5.out 2>&1 & wait
nohup python main.py -d wsn_ds -m SupAE -tbs 128 -vbs 1 -di 17 -lr 0.001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log6.out 2>&1 & wait
nohup python main.py -d wsn_ds -m FedHome -tbs 128 -vbs 1 -di 17 -lr 0.001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log8.out 2>&1 & wait
nohup python main.py -d wsn_ds -m FedGH -tbs 128 -vbs 1 -di 17 -lr 0.001 -ep 10 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log9.out 2>&1 & 
nohup python main.py -d wsn_ds -m DualLossAE -tbs 128 -vbs 1 -di 17 -lr 0.001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log7.out 2>&1 &

# 2. UNSW:
nohup python -u initialize_env.py -data unsw  > logger/unsw_init.out 2>&1 &

# Thí nghiệm limited attack
nohup python main.py -d unsw -m AE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m DAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m FedMSE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg FedMSE -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SupAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 10 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 5 > logger/unsw_log.out 2>&1 & wait
nohup python main.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 10 > logger/unsw_log.out 2>&1 & wait
nohup python main.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 > logger/unsw_log.out 2>&1 & wait
nohup python main.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 2 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m PTLAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 5 > logger/unsw_log1.out 2>&1 & wait
nohup python main.py -d unsw -m PTLAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 10 > logger/unsw_log2.out 2>&1 & 
nohup python main.py -d unsw -m PTLAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 > logger/unsw_log3.out 2>&1 & wait
nohup python main.py -d unsw -m PTLAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 2 > logger/unsw_log4.out 2>&1 &

nohup python evaluate.py -d unsw -m AE -tbs 128 -vbs 128 -di 196 -ep 1000 -tm 1.2 --model_dir saved_models/AE/average/unsw --log_csv logs/AE/unsw_AE_ > logger/unsw_AE_1000_1.2.log 2>&1 &
nohup python evaluate.py -d unsw -m DAE -tbs 128 -vbs 128 -di 196 -ep 1000 -tm 1.5 --model_dir saved_models/DAE/average/unsw --log_csv logs/DAE/unsw_DAE_ > logger/unsw_DAE_1000_1.5.log 2>&1 &
nohup python evaluate.py -d unsw -m SAE -tbs 128 -vbs 128 -di 196 -ep 1000 -tm 2.1 --model_dir saved_models/SAE/average/unsw --log_csv logs/SAE/unsw_SAE_ > logger/unsw_SAE_1000_2.1.log 2>&1 &
nohup python evaluate.py -d unsw -m FedMSE -tbs 128 -vbs 128 -di 196 -ep 1000 -tm 0.5 --model_dir saved_models/FedMSE/FedMSE/unsw --log_csv logs/FedMSE/unsw_FedMSE_ > logger/unsw_FedMSE_1000_0.5.log 2>&1 &
nohup python evaluate.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 0.6 --model_dir saved_models/DualLossAE2old/average/unsw --log_csv logs/DualLossAE2old/unsw_DualLossAE > logger/unsw_DualLossAE2old_1000_0.6.log 2>&1 &
nohup python evaluate.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 0.5 --model_dir saved_models/DualLossAE5old/average/unsw --log_csv logs/DualLossAE5old/unsw_DualLossAE > logger/unsw_DualLossAE5old_1000_0.5.log 2>&1 &
nohup python evaluate.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 0.4 --model_dir saved_models/DualLossAE10old/average/unsw --log_csv logs/DualLossAE10old/unsw_DualLossAE > logger/unsw_DualLossAE10old_1000_0.4.log 2>&1 &
nohup python evaluate.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 0.2 --model_dir saved_models/DualLossAE20old/average/unsw --log_csv logs/DualLossAE20old/unsw_DualLossAE > logger/unsw_DualLossAE20old_1000_0.2.log 2>&1 &
nohup python evaluate.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 0.6 --model_dir saved_models/DualLossAE2/average/unsw --log_csv logs/DualLossAE2/unsw_DualLossAE > logger/unsw_DualLossAE2_1000_0.6.log 2>&1 &
nohup python evaluate.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 1.1 --model_dir saved_models/DualLossAE5/average/unsw --log_csv logs/DualLossAE5/unsw_DualLossAE > logger/unsw_DualLossAE5_1000_1.1.log 2>&1 &
nohup python evaluate.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -ep 400 -tm 1 --model_dir saved_models/DualLossAE5/average/unsw --log_csv logs/unsw_DualLossAE > logger/unsw_DualLossAE5_400_1.log 2>&1 &
nohup python evaluate.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 1.4 --model_dir saved_models/DualLossAE20/average/unsw --log_csv logs/DualLossAE20/unsw_DualLossAE > logger/unsw_DualLossAE20_1000_1.4.log 2>&1 &

# Thí nghiệm unseen attack
nohup python main.py -d unsw -m PTLAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log5.out 2>&1 & wait
nohup python main.py -d unsw -m SupAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log6.out 2>&1 & wait
nohup python main.py -d unsw -m SAE -tbs 128 -vbs 1 -di 196 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 -et non_iid_dir -da 1.0  > logger/unsw_log7.out 2>&1 &
nohup python main.py -d unsw -m FedHome -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log8.out 2>&1 &
nohup python main.py -d unsw -m FedGH -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 10 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log9.out 2>&1 &
nohup python main.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log7.out 2>&1 &

nohup python evaluate.py -d unsw -m PTLAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 1 -mc 20 -et non_iid_dir --model_dir saved_models/PTLAE/20/average/unsw --log_csv logs/unseen/unsw_PTLAE_ > logger/unsw_PTLAE_1000.log 2>&1 & wait
nohup python evaluate.py -d unsw -m PTLAE -tbs 128 -vbs 1 -di 196 -ep 300 -tm 1 -mc 20 -et non_iid_dir --model_dir saved_models/PTLAE/20/average/unsw --log_csv logs/unseen/unsw_PTLAE_ > logger/unsw_PTLAE_300.log 2>&1 & wait
nohup python evaluate.py -d unsw -m PTLAE -tbs 128 -vbs 1 -di 196 -ep 150 -tm 1 -mc 20 -et non_iid_dir --model_dir saved_models/PTLAE/20/average/unsw --log_csv logs/unseen/unsw_PTLAE_ > logger/unsw_PTLAE_150.log 2>&1 & wait
nohup python evaluate.py -d unsw -m SupAE -tbs 128 -vbs 1 -di 196 -ep 200 -tm 1 -mc 20 -et non_iid_dir --model_dir saved_models/SupAE/20/average/unsw --log_csv logs/unseen/unsw_SupAE_ > logger/unsw_SupAE_200.log 2>&1 & wait
nohup python evaluate.py -d unsw -m SupAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 1 -mc 20 -et non_iid_dir --model_dir saved_models/SupAE/20/average/unsw --log_csv logs/unsw_SupAE_ > logger/unsw_SupAE_1000.log 2>&1 &
nohup python evaluate.py -d unsw -m SupAE -tbs 128 -vbs 1 -di 196 -ep 4000 -tm 1 -mc 20 -et non_iid_dir --model_dir saved_models/SupAE/20/average/unsw --log_csv logs/unseen/unsw_SupAE_ > logger/unsw_SupAE_4000.log 2>&1 &
nohup python evaluate.py -d unsw -m FedHome -tbs 128 -vbs 1 -di 196 -ep 20 -tm 0 -mc 20 -et non_iid_dir --model_dir saved_models/FedHome/20/average/unsw --log_csv logs/unsw_FedHome_ > logger/unsw_FedHome_20.log 2>&1 &
nohup python evaluate.py -d unsw -m FedGH -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 0 -mc 20 -et non_iid_dir --model_dir saved_models/FedGH/20/average/unsw --log_csv logs/unsw_FedGH_ > logger/unsw_FedGH_10.log 2>&1 &
nohup python evaluate.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 1 -mc 20 -et non_iid_dir --model_dir saved_models/DualLossAE/20/average/unsw --log_csv logs/unsw_DualLossAE_ > logger/unsw_DualLossAE_1000.log 2>&1 &

nohup python main.py -d unsw -m PTLAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log5.out 2>&1 
nohup python evaluate.py -d unsw -m PTLAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 1 -mc 20 -et non_iid_dir --model_dir saved_models/PTLAE/20/average/unsw --log_csv logs/unsw_PTLAE_ > logger/unsw_PTLAE_1000.log 2>&1 
nohup python evaluate.py -d unsw -m PTLAE -tbs 128 -vbs 1 -di 196 -ep 300 -tm 1 -mc 20 -et non_iid_dir --model_dir saved_models/PTLAE/20/average/unsw --log_csv logs/unsw_PTLAE_ > logger/unsw_PTLAE_300.log 2>&1 
nohup python main.py -d unsw -m SupAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log6.out 2>&1 
nohup python evaluate.py -d unsw -m SupAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 1 -mc 20 -et non_iid_dir --model_dir saved_models/SupAE/20/average/unsw --log_csv logs/unsw_SupAE_ > logger/unsw_SupAE_1000.log 2>&1 
nohup python main.py -d unsw -m FedHome -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log8.out 2>&1 
nohup python evaluate.py -d unsw -m FedHome -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 0 -mc 20 -et non_iid_dir --model_dir saved_models/FedHome/20/average/unsw --log_csv logs/unsw_FedHome_ > logger/unsw_FedHome_1000.log 2>&1 
nohup python main.py -d unsw -m FedGH -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log9.out 2>&1 
nohup python evaluate.py -d unsw -m FedGH -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 0 -mc 20 -et non_iid_dir --model_dir saved_models/FedGH/20/average/unsw --log_csv logs/unsw_FedGH_ > logger/unsw_FedGH_1000.log 2>&1 



# Thí nghiệm unseen attack cũ
nohup python main.py -d unsw -m PTLAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 200 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 10 -at 1 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 2 -at 1 > logger/unsw_log.out 2>&1 &
nohup python evaluate_attack_type.py -d unsw -m AE -tbs 128 -vbs 128 -di 196 -ep 1000 -tm 1.2 --model_dir saved_models/AE/average/unsw --log_csv logs/AE/unsw_AE_ > evaluate_results/attacktype_unsw_AE_1000_1.2.log 2>&1 &
nohup python evaluate_attack_type.py -d unsw -m DAE -tbs 128 -vbs 128 -di 196 -ep 1000 -tm 1.5 --model_dir saved_models/DAE/average/unsw --log_csv logs/DAE/unsw_DAE_ > evaluate_results/attacktype_unsw_DAE_1000_1.5.log 2>&1 &
nohup python evaluate_attack_type.py -d unsw -m SAE -tbs 128 -vbs 128 -di 196 -ep 1000 -tm 2.1 --model_dir saved_models/SAE/average/unsw --log_csv logs/SAE/unsw_SAE_ > evaluate_results/attacktype_unsw_SAE_1000_2.1.log 2>&1 &
nohup python evaluate_attack_type.py -d unsw -m DualLossAE -tbs 128 -vbs 1 -di 196 -ep 1000 -tm 0.6 --model_dir saved_models/DualLossAEbytype/average/unsw --log_csv logs/DualLossAEbytype/unsw_DualLossAE > evaluate_results/attacktype_unsw_DualLossAE2_1000_0.6.log 2>&1 &

# Thí nghiệm với tấn công label flipping và gaussian noise
nohup python main.py -d unsw -m AE -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m DAE -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SAE -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SAE1 -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SDAE -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SDAE1 -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m AE -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m DAE -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SAE -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SAE1 -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SDAE -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SDAE1 -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m AE -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m DAE -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SAE -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SAE1 -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SDAE -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &
nohup python main.py -d unsw -m SDAE1 -tbs 128 -vbs 128 -di 196 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/unsw_log.out 2>&1 &

# 3. CIC_IDS:
nohup python -u initialize_env.py -data cic_ids  > logger/cic_ids_init.out 2>&1 &

nohup python main.py -d cic_ids -m AE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 4000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0  > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m DAE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 4000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0  > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SAE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 4000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0  > logger/cic_ids_log2.out 2>&1 &
nohup python main.py -d cic_ids -m FedMSE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 4000 -agg FedMSE -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0  > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SupAE -tbs 128 -vbs 1 -di 80 -lr 0.001 -ep 4000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 10  > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -lr 0.001 -ep 4000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 5  > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -lr 0.001 -ep 4000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 10  > logger/cic_ids_log2.out 2>&1 &
nohup python main.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -lr 0.001 -ep 4000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 20  > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -lr 0.001 -ep 4000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 2  > logger/cic_ids_log.out 2>&1 &

nohup python evaluate.py -d cic_ids -m AE -tbs 128 -vbs 128 -di 80 -ep 4000 -tm 0.0 --model_dir saved_models/AE/average/cic_ids  --log_csv logs/AE/cic_ids_AE_ > logger/cic_ids_logAE.out 2>&1 
nohup python evaluate.py -d cic_ids -m DAE -tbs 128 -vbs 128 -di 80 -ep 4000 -tm 0.4 --model_dir saved_models/DAE/average/cic_ids  --log_csv logs/DAE/cic_ids_DAE_ > logger/cic_ids_logDAE.out 2>&1 
nohup python evaluate.py -d cic_ids -m SAE -tbs 128 -vbs 128 -di 80 -ep 4000 -tm 3.0 --model_dir saved_models/SAE/average/cic_ids  --log_csv logs/SAE/cic_ids_SAE_ > logger/cic_ids_logSAE.out 2>&1 
nohup python evaluate.py -d cic_ids -m FedMSE -tbs 128 -vbs 128 -di 80 -ep 4000 -tm 3.0 --model_dir saved_models/FedMSE/FedMSE/cic_ids  --log_csv logs/FedMSE/cic_ids_FedMSE_ > logger/cic_ids_logFeDMSE.out 2>&1 
nohup python evaluate.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -ep 4000 -tm 0 --model_dir saved_models/DualLossAE2old/average/cic_ids  --log_csv logs/DualLossAE2old/cic_ids_DualLossAE > logger/cic_ids_logDL2old.out 2>&1 
nohup python evaluate.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -ep 4000 -tm 0 --model_dir saved_models/DualLossAE5old/average/cic_ids  --log_csv logs/DualLossAE5old/cic_ids_DualLossAE > logger/cic_ids_logDL5old.out 2>&1 
nohup python evaluate.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -ep 4000 -tm 0 --model_dir saved_models/DualLossAE10old/average/cic_ids  --log_csv logs/DualLossAE10old/cic_ids_DualLossAE > logger/cic_ids_logDL10old.out 2>&1 
nohup python evaluate.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -ep 4000 -tm 0.7 --model_dir saved_models/DualLossAE20old/average/cic_ids  --log_csv logs/DualLossAE20old/cic_ids_DualLossAE > logger/cic_ids_logDL20old.out 2>&1 
nohup python evaluate.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -ep 4000 -tm 0.7 --model_dir saved_models/DualLossAE2/average/cic_ids  --log_csv logs/DualLossAE2/cic_ids_DualLossAE > logger/cic_ids_logDL2.out 2>&1
nohup python evaluate.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -ep 4000 -tm 0.7 --model_dir saved_models/DualLossAE5/average/cic_ids  --log_csv logs/DualLossAE5/cic_ids_DualLossAE > logger/cic_ids_logDL5.out 2>&1 
nohup python evaluate.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -ep 4000 -tm 0.7 --model_dir saved_models/DualLossAE10/average/cic_ids  --log_csv logs/DualLossAE10/cic_ids_DualLossAE > logger/cic_ids_logDL10.out 2>&1 
nohup python evaluate.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -ep 4000 -tm 0.7 --model_dir saved_models/DualLossAE20/average/cic_ids  --log_csv logs/DualLossAE20/cic_ids_DualLossAE > logger/cic_ids_logDL20.out 2>&1 

nohup python main.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -lr 0.0001 -ep 4000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 2 -at 1 > logger/cic_ids_log.out 2>&1 &
nohup python evaluate_attack_type.py -d cic_ids -m AE -tbs 128 -vbs 128 -di 80 -ep 4000 -tm 0.0 --model_dir saved_models/AE/average/cic_ids  --log_csv logs/AE/cic_ids_AE_ > evaluate_results/attacktype_cic_ids_AE_4000_0.log 2>&1 &
nohup python evaluate_attack_type.py -d cic_ids -m DAE -tbs 128 -vbs 128 -di 80 -ep 4000 -tm 0.4 --model_dir saved_models/DAE/average/cic_ids  --log_csv logs/DAE/cic_ids_DAE_ > evaluate_results/attacktype_cic_ids_DAE_4000_0.4.log 2>&1 &
nohup python evaluate_attack_type.py -d cic_ids -m SAE -tbs 128 -vbs 128 -di 80 -ep 4000 -tm 3.0 --model_dir saved_models/SAE/average/cic_ids  --log_csv logs/SAE/cic_ids_SAE_ > evaluate_results/attacktype_cic_ids_SAE_4000_3.log 2>&1 &
nohup python evaluate_attack_type.py -d cic_ids -m DualLossAE -tbs 128 -vbs 1 -di 80 -ep 4000 -tm 0 --model_dir saved_models/DualLossAEbytype1/average/cic_ids  --log_csv logs/DualLossAEbytype1/cic_ids_DualLossAE > evaluate_results/attacktype1_cic_ids_DualLossAE_4000_0.log 2>&1 &

nohup python main.py -d cic_ids -m PTLAE -tbs 128 -vbs 1 -di 80 -lr 0.001 -ep 10 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 1 -mc 20 -et non_iid_dir -da 1.0 > logger/unsw_log5.out 2>&1 &


nohup python main.py -d cic_ids -m AE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m DAE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SAE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SAE1 -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SDAE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SDAE1 -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m AE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m DAE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SAE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SAE1 -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SDAE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SDAE1 -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m AE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m DAE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SAE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SAE1 -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SDAE -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &
nohup python main.py -d cic_ids -m SDAE1 -tbs 128 -vbs 128 -di 80 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/cic_ids_log.out 2>&1 &

# 4. NB_IOT:
nohup python -u initialize_env.py -data nb_iot  > logger/nb_iot_init.out 2>&1 &

nohup python main.py -d nb_iot -m AE -tbs 128 -vbs 128 -di 115 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m DAE -tbs 128 -vbs 128 -di 115 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m SAE -tbs 128 -vbs 128 -di 115 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m FedMSE -tbs 128 -vbs 128 -di 115 -lr 0.0001 -ep 2000 -agg FedMSE -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m SupAE -tbs 128 -vbs 1 -di 115 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 10 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m DualLossAE -tbs 128 -vbs 1 -di 115 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 5 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m DualLossAE -tbs 128 -vbs 1 -di 115 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 10 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m DualLossAE -tbs 128 -vbs 1 -di 115 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 20 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m DualLossAE -tbs 128 -vbs 1 -di 115 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 2 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m PTLAE -tbs 128 -vbs 1 -di 115 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 2 > logger/nb_iot_log1.out 2>&1 &
nohup python main.py -d nb_iot -m PTLAE -tbs 128 -vbs 1 -di 115 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 5 > logger/nb_iot_log2.out 2>&1 &
nohup python main.py -d nb_iot -m PTLAE -tbs 128 -vbs 1 -di 115 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 10 > logger/nb_iot_log3.out 2>&1 &
nohup python main.py -d nb_iot -m PTLAE -tbs 128 -vbs 1 -di 115 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 20 > logger/nb_iot_log4.out 2>&1 &

nohup python evaluate.py -d nb_iot -m AE -tbs 128 -vbs 128 -di 115 -ep 2000 -tm 2 --model_dir saved_models/AE/average/nb_iot --log_csv logs/AE/nb_iot_AE_ > logger/nb_iot_AE_2000_2.log 2>&1
nohup python evaluate.py -d nb_iot -m DAE -tbs 128 -vbs 128 -di 115 -ep 2000 -tm 1.5 --model_dir saved_models/DAE/average/nb_iot --log_csv logs/DAE/nb_iot_DAE_ > logger/nb_iot_DAE_2000_1.5.log 2>&1
nohup python evaluate.py -d nb_iot -m SAE -tbs 128 -vbs 128 -di 115 -ep 2000 -tm 3 --model_dir saved_models/SAE/average/nb_iot --log_csv logs/SAE/nb_iot_SAE_ > logger/nb_iot_SAE_2000_3.log 2>&1
nohup python evaluate.py -d nb_iot -m FedMSE -tbs 128 -vbs 128 -di 115 -ep 2000 -tm 3 --model_dir saved_models/FedMSE/FedMSE/nb_iot --log_csv logs/FedMSE/nb_iot_FedMSE_ > logger/nb_iot_FedMSE_2000_3.log 2>&1
nohup python evaluate.py -d nb_iot -m DualLossAE -tbs 128 -vbs 1 -di 115 -ep 2000 -tm 2 --model_dir saved_models/DualLossAE2old/average/nb_iot --log_csv logs/DualLossAE2old/nb_iot_DualLossAE > logger/nb_iot_DualLossAE2old_2000_2.log 2>&1
nohup python evaluate.py -d nb_iot -m DualLossAE -tbs 128 -vbs 1 -di 115 -ep 2000 -tm 0.2 --model_dir saved_models/DualLossAE5old/average/nb_iot --log_csv logs/DualLossAE5old/nb_iot_DualLossAE > logger/nb_iot_DualLossAE5old_2000_0.2.log 2>&1
nohup python evaluate.py -d nb_iot -m DualLossAE -tbs 128 -vbs 1 -di 115 -ep 2000 -tm 0.1 --model_dir saved_models/DualLossAE10old/average/nb_iot --log_csv logs/DualLossAE10old/nb_iot_DualLossAE > logger/nb_iot_DualLossAE10old_2000_0.1.log 2>&1
nohup python evaluate.py -d nb_iot -m DualLossAE -tbs 128 -vbs 1 -di 115 -ep 2000 -tm 3 --model_dir saved_models/DualLossAE20old/average/nb_iot --log_csv logs/DualLossAE20old/nb_iot_DualLossAE > logger/nb_iot_DualLossAE20old_2000_3.log 2>&1
nohup python evaluate.py -d nb_iot -m DualLossAE -tbs 128 -vbs 1 -di 115 -ep 2000 -tm 2 --model_dir saved_models/DualLossAE2/average/nb_iot --log_csv logs/DualLossAE2/nb_iot_DualLossAE > logger/nb_iot_DualLossAE2_2000_2.log 2>&1
nohup python evaluate.py -d nb_iot -m DualLossAE -tbs 128 -vbs 1 -di 115 -ep 2000 -tm 2 --model_dir saved_models/DualLossAE5/average/nb_iot --log_csv logs/DualLossAE5/nb_iot_DualLossAE > logger/nb_iot_DualLossAE5_2000_2.log 2>&1
nohup python evaluate.py -d nb_iot -m DualLossAE -tbs 128 -vbs 1 -di 115 -ep 2000 -tm 2 --model_dir saved_models/DualLossAE10/average/nb_iot --log_csv logs/DualLossAE10/nb_iot_DualLossAE > logger/nb_iot_DualLossAE10_2000_2.log 2>&1
nohup python evaluate.py -d nb_iot -m DualLossAE -tbs 128 -vbs 1 -di 115 -ep 2000 -tm 3 --model_dir saved_models/DualLossAE20/average/nb_iot --log_csv logs/DualLossAE20/nb_iot_DualLossAE > logger/nb_iot_DualLossAE20_2000_3.log 2>&1

# Thí nghiệm unseen attack
nohup python main.py -d nb_iot -m PTLAE -tbs 128 -vbs 1 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 20 -et non_iid_dir > logger/nb_iot_log1.out 2>&1 & wait
nohup python main.py -d nb_iot -m SupAE -tbs 128 -vbs 1 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 20 -et non_iid_dir > logger/nb_iot_log2.out 2>&1 & wait
nohup python main.py -d nb_iot -m DuallossAE -tbs 128 -vbs 1 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 20 -et non_iid_dir > logger/nb_iot_log3.out 2>&1 & 


nohup python main.py -d nb_iot -m AE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m DAE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log1.out 2>&1 &
nohup python main.py -d nb_iot -m SAE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log2.out 2>&1 &
nohup python main.py -d nb_iot -m FedMSE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg FedMSE -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log3.out 2>&1 &
nohup python main.py -d nb_iot -m SAE1 -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log3.out 2>&1 &


nohup python evaluate.py -d nb_iot -m AE -tbs 128 -vbs 128 -di 115 -ep 500 -tm 2 --model_dir saved_models/conf_5/AE/average/nb_iot --log_csv logs/conf_5/nb_iot_AE_ > logger/conf_5/nb_iot_AE_2.log 2>&1 &
nohup python evaluate.py -d nb_iot -m DAE -tbs 128 -vbs 128 -di 115 -ep 500 -tm 4 --model_dir saved_models/conf_5/DAE/average/nb_iot --log_csv logs/conf_5/nb_iot_DAE_ > logger/conf_5/nb_iot_DAE_4.log 2>&1 &
nohup python evaluate.py -d nb_iot -m SAE -tbs 128 -vbs 128 -di 115 -ep 500 -tm 4 --model_dir saved_models/conf_5/SAE/average/nb_iot --log_csv logs/conf_5/nb_iot_SAE_ > logger/conf_5/nb_iot_SAE_4.log 2>&1 &
nohup python evaluate.py -d nb_iot -m SAE1 -tbs 128 -vbs 128 -di 115 -ep 500 -tm 4 --model_dir saved_models/conf_5/SAE1/average/nb_iot --log_csv logs/conf_5/nb_iot_SAE1_ > logger/conf_5/nb_iot_SAE1_4.log 2>&1 &
nohup python evaluate.py -d nb_iot -m FedMSE -tbs 128 -vbs 128 -di 115 -ep 500 -tm 4 --model_dir saved_models/conf_5/FedMSE/FedMSE/nb_iot --log_csv logs/conf_5/nb_iot_FedMSE_ > logger/conf_5/nb_iot_FedMSE_4.log 2>&1 &



nohup python main.py -d nb_iot -m SDAE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log4.out 2>&1 &
nohup python main.py -d nb_iot -m SDAE1 -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log5.out 2>&1 &
nohup python main.py -d nb_iot -m AE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m DAE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m SAE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m SAE1 -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m SDAE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m SDAE1 -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m AE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m DAE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m SAE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m SAE1 -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m SDAE -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &
nohup python main.py -d nb_iot -m SDAE1 -tbs 128 -vbs 128 -di 115 -lr 0.001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nb_iot_log.out 2>&1 &


# 5. CTU13_08:
nohup python -u initialize_env.py -data ctu13_08  > logger/ctu13_08_init.out 2>&1 &


nohup python main.py -d ctu13_08 -m PTLAE -tbs 256 -vbs 1 -di 40 -lr 0.0001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 20 -et non_iid_dir > logger/ctu13_08_log1.out 2>&1 & 


nohup python main.py -d ctu13_08 -m AE -tbs 256 -vbs 256 -di 40 -lr 0.0001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SAE -tbs 256 -vbs 256 -di 40 -lr 0.0001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m DAE -tbs 256 -vbs 256 -di 40 -lr 0.0001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m FedMSE -tbs 256 -vbs 256 -di 40 -lr 0.0001 -ep 2000 -agg FedMSE -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SupAE -tbs 256 -vbs 1 -di 40 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 5 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m DualLossAE -tbs 256 -vbs 1 -di 40 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 5 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m DualLossAE -tbs 256 -vbs 1 -di 40 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 10 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m DualLossAE -tbs 256 -vbs 1 -di 40 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 20 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m DualLossAE -tbs 256 -vbs 1 -di 40 -lr 0.0001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0.3 -mc 2 > logger/ctu13_08_log.out 2>&1 & 

nohup python evaluate.py -d ctu13_08 -m AE -tbs 256 -vbs 256 -di 40 -ep 2000 -tm 3 --model_dir saved_models/AE/average/ctu13_08 --log_csv logs/AE/ctu13_08_AE_ > logger/ctu13_08_AE_2000_3.log 2>&1
nohup python evaluate.py -d ctu13_08 -m DAE -tbs 256 -vbs 256 -di 40 -ep 2000 -tm 3 --model_dir saved_models/DAE/average/ctu13_08 --log_csv logs/DAE/ctu13_08_DAE_ > logger/ctu13_08_DAE_2000_3.log 2>&1
nohup python evaluate.py -d ctu13_08 -m SAE -tbs 256 -vbs 256 -di 40 -ep 2000 -tm 1.6 --model_dir saved_models/SAE/average/ctu13_08 --log_csv logs/SAE/ctu13_08_SAE_ > logger/ctu13_08_SAE_2000_1.6.log 2>&1
nohup python evaluate.py -d ctu13_08 -m FedMSE -tbs 256 -vbs 256 -di 40 -ep 2000 -tm 1.8 --model_dir saved_models/FedMSE/FedMSE/ctu13_08 --log_csv logs/FedMSE/ctu13_08_FedMSE_ > logger/ctu13_08_FedMSE_2000_1.8.log 2>&1
nohup python evaluate.py -d ctu13_08 -m DualLossAE -tbs 256 -vbs 1 -di 40 -ep 2000 -tm 0.3 --model_dir saved_models/DualLossAE2old/average/ctu13_08 --log_csv logs/DualLossAE2old/ctu13_08_DualLossAE > logger/ctu13_08_DualLossAE2old_2000_0.3.log 2>&1
nohup python evaluate.py -d ctu13_08 -m DualLossAE -tbs 256 -vbs 1 -di 40 -ep 2000 -tm 0.9 --model_dir saved_models/DualLossAE5old/average/ctu13_08 --log_csv logs/DualLossAE5old/ctu13_08_DualLossAE > logger/ctu13_08_DualLossAE5old_2000_0.9.log 2>&1
nohup python evaluate.py -d ctu13_08 -m DualLossAE -tbs 256 -vbs 1 -di 40 -ep 2000 -tm 3 --model_dir saved_models/DualLossAE10old/average/ctu13_08 --log_csv logs/DualLossAE10old/ctu13_08_DualLossAE > logger/ctu13_08_DualLossAE10old_2000_3.log 2>&1
nohup python evaluate.py -d ctu13_08 -m DualLossAE -tbs 256 -vbs 1 -di 40 -ep 2000 -tm 0.5 --model_dir saved_models/DualLossAE20old/average/ctu13_08 --log_csv logs/DualLossAE20old/ctu13_08_DualLossAE > logger/ctu13_08_DualLossAE20old_2000_0.5.log 2>&1
nohup python evaluate.py -d ctu13_08 -m DualLossAE -tbs 256 -vbs 1 -di 40 -ep 2000 -tm 0.8 --model_dir saved_models/DualLossAE2/average/ctu13_08 --log_csv logs/DualLossAE2/ctu13_08_DualLossAE > logger/ctu13_08_DualLossAE2_2000_0.8.log 2>&1
nohup python evaluate.py -d ctu13_08 -m DualLossAE -tbs 256 -vbs 1 -di 40 -ep 2000 -tm 1.1 --model_dir saved_models/DualLossAE5/average/ctu13_08 --log_csv logs/DualLossAE5/ctu13_08_DualLossAE > logger/ctu13_08_DualLossAE5_2000_1.1.log 2>&1
nohup python evaluate.py -d ctu13_08 -m DualLossAE -tbs 256 -vbs 1 -di 40 -ep 2000 -tm 1.1 --model_dir saved_models/DualLossAE10/average/ctu13_08 --log_csv logs/DualLossAE10/ctu13_08_DualLossAE > logger/ctu13_08_DualLossAE10_2000_1.1.log 2>&1
nohup python evaluate.py -d ctu13_08 -m DualLossAE -tbs 256 -vbs 1 -di 40 -ep 2000 -tm 0.6 --model_dir saved_models/DualLossAE20/average/ctu13_08 --log_csv logs/DualLossAE20/ctu13_08_DualLossAE > logger/ctu13_08_DualLossAE20_2000_0.6.log 2>&1

nohup python main.py -d ctu13_08 -m AE -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m DAE -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SAE -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SAE1 -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SDAE -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SDAE1 -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m AE -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m DAE -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SAE -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SAE1 -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SDAE -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SDAE1 -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m AE -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m DAE -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SAE -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SAE1 -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SDAE -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &
nohup python main.py -d ctu13_08 -m SDAE1 -tbs 256 -vbs 256 -di 40 -lr 0.01 -ep 700 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/ctu13_08_log.out 2>&1 &

#6. NSL_KDD:
nohup python -u initialize_env.py -data nsl_kdd  > logger/nsl_kdd_init.out 2>&1 &

nohup python main.py -d nsl_kdd -m AE -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m DAE -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m SAE -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m SAE1 -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m SDAE -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m SDAE1 -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt label_flipping -pw 0 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m AE -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m DAE -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m SAE -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m SAE1 -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m SDAE -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m SDAE1 -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt label_flipping -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m AE -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m DAE -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m SAE -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m SAE1 -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m SDAE -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &
nohup python main.py -d nsl_kdd -m SDAE1 -tbs 32 -vbs 32 -di 122 -lr 0.0001 -ep 500 -agg average -nt gaussian_noise -pw 5 -pr 0.5 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 0 > logger/nsl_kdd_log.out 2>&1 &


nohup python main.py -d nsl_kdd -m PTLAE -tbs 32 -vbs 1 -di 122 -lr 0.0001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 20 -et non_iid_dir > logger/nsl_kdd_log1.out 2>&1 & 
nohup python evaluate.py -d nsl_kdd -m PTLAE -tbs 32 -vbs 1 -di 122 -ep 300 -tm 1 -mc 20 -et non_iid_dir --model_dir saved_models/PTLAE/20/average/nsl_kdd --log_csv logs/nsl_kdd_PTLAE_ > logger/nsl_kdd_PTLAE_300.log 2>&1 
nohup python main.py -d nsl_kdd -m SupAE -tbs 32 -vbs 1 -di 122 -lr 0.0001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 -et non_iid_dir -da 1.0 > logger/nsl_kdd_log6.out 2>&1 
nohup python evaluate.py -d nsl_kdd -m SupAE -tbs 32 -vbs 1 -di 122 -ep 1000 -tm 1 -mc 20 -et non_iid_dir --model_dir saved_models/SupAE/20/average/nsl_kdd --log_csv logs/nsl_kdd_SupAE_ > logger/nsl_kdd_SupAE_1000.log 2>&1 
nohup python main.py -d nsl_kdd -m FedHome -tbs 32 -vbs 1 -di 122 -lr 0.0001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 20 -et non_iid_dir -da 1.0 > logger/nsl_kdd_log8.out 2>&1 
nohup python evaluate.py -d nsl_kdd -m FedHome -tbs 32 -vbs 1 -di 122 -ep 1000 -tm 0 -mc 20 -et non_iid_dir --model_dir saved_models/FedHome/20/average/nsl_kdd --log_csv logs/nsl_kdd_FedHome_ > logger/nsl_kdd_FedHome_1000.log 2>&1 
nohup python main.py -d nsl_kdd -m FedGH -tbs 32 -vbs 1 -di 122 -lr 0.0001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 0 -mc 20 -et non_iid_dir -da 1.0 > logger/nsl_kdd_log9.out 2>&1 
nohup python evaluate.py -d nsl_kdd -m FedGH -tbs 32 -vbs 1 -di 122 -ep 1000 -tm 0 -mc 20 -et non_iid_dir --model_dir saved_models/FedGH/20/average/nsl_kdd --log_csv logs/nsl_kdd_FedGH_ > logger/nsl_kdd_FedGH_1000.log 2>&1 


# 7. TON_IOT:
nohup python -u initialize_env.py -data ton_iot_network  > logger/ton_iot_init.out 2>&1 &
nohup python main.py -d ton_iot_network -m AE -tbs 128 -vbs 1 -di 40 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 0.5 -cs 1 -tm 1 -mc 0 > logger/ton_iot_log.out 2>&1 & 
nohup python main.py -d ton_iot_network -m SAE -tbs 128 -vbs 1 -di 40 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 0.5 -cs 1 -tm 1 -mc 0 > logger/ton_iot_log.out 2>&1 & 
nohup python main.py -d ton_iot_network -m SupAE -tbs 128 -vbs 1 -di 40 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 0.5 -cs 1 -tm 1 -mc 20 > logger/ton_iot_log.out 2>&1 &
nohup python main.py -d ton_iot_network -m DAE -tbs 128 -vbs 1 -di 16 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 0.5 -cs 1 -tm 1 -mc 0 > logger/ton_iot_log.out 2>&1 & 
nohup python main.py -d ton_iot_network -m SupAE -tbs 128 -vbs 1 -di 16 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 0.5 -cs 1 -tm 1 -mc 20 > logger/ton_iot_log.out 2>&1 & 
nohup python main.py -d ton_iot_network -m DualLossAE -tbs 128 -vbs 1 -di 40 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 0.5 -cs 1 -tm 1 -mc 2 > logger/ton_iot_log.out 2>&1 & 
nohup python main.py -d ton_iot_network -m DualLossAE -tbs 128 -vbs 1 -di 40 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 0.5 -cs 1 -tm 1 -mc 2 > logger/ton_iot_log.out 2>&1 & 



# CIC IOT:

nohup python -u initialize_env.py -data cic_iot  > logger/cic_iot_init.out 2>&1 &

nohup python main.py -d cic_iot -m PTLAE -tbs 128 -vbs 1 -di 46 -lr 0.0001 -ep 1000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 -et non_iid_dir -da 1.0 > logger/cic_iot_PTLAE_log20.out   2>&1 &

