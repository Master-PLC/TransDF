#!/bin/bash
MAX_JOBS=12
GPUS=(0 3 4 7)
TOTAL_GPUS=${#GPUS[@]}

get_gpu_allocation(){
    local job_number=$1
    # Calculate which GPU to allocate based on the job number
    local gpu_id=${GPUS[$((job_number % TOTAL_GPUS))]}
    echo $gpu_id
}

check_jobs(){
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

job_number=0

DATA_ROOT=./dataset
EXP_NAME=validation
seed=2023
des='FBM_L'

model_name=FBM_L
auxi_mode=fft_ot

# datasets=(ETTh1 ETTh2 ETTm1 ETTm2 ECL Traffic Weather PEMS03 PEMS08)
# datasets=(ETTh1_PCA ETTh2_PCA ETTm1_PCA ETTm2_PCA Weather_PCA)
datasets=(ETTh1 ETTm1)
# datasets=(ETTh1)



# hyper-parameters
dst=ETTh1

# pl_list=(96 192 336 720)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(96 192 336 720)
lbd_list=(0.8 0.6 0.4 0.2 0.0)
lr_list=(0.001)
mask_factor_list=(0.01)
distance_list=(fft_1norm fft_2norm time)
joint_forecast_list=(0 1)


decomposition=0
normalize=1
reg_sk=0.1
auxi_loss=None
ot_type=sinkhorn
lradj=type1
train_epochs=10
patience=3
batch_size=32
test_batch_size=1

rerun=0

for mask_factor in ${mask_factor_list[@]}; do
for distance in ${distance_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}
    OUTPUT_DIR="./results_FFTOT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --decomposition ${decomposition} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 2
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done





# hyper-parameters
dst=ETTh2

# pl_list=(96 192 336 720)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(96 192 336 720)
lbd_list=(1.0 0.8 0.6 0.4 0.2 0.0)
lr_list=(0.001)
auxi_loss_list=(None)


lradj=type1
train_epochs=10
patience=3
batch_size=32
test_batch_size=1

rerun=0



for auxi_loss in ${auxi_loss_list[@]}; do
for use_weights in ${use_weights_list[@]}; do
for reinit in ${reinit_list[@]}; do
for pca_dim in ${pca_dim_list[@]}; do
for rank_ratio in ${rank_ratio_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_FFTOT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh2.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh2 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --decomposition ${decomposition} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 2
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done






# hyper-parameters
dst=ETTm1

# pl_list=(96 192 336 720)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(96 192 336 720)
lbd_list=(1.0 0.8 0.6 0.4 0.2 0.0)
lr_list=(0.001)
mask_factor_list=(0.01 0.1)
distance_list=(fft_2norm time)
joint_forecast_list=(0 1)


decomposition=0
normalize=1
reg_sk=0.1
auxi_loss=None
ot_type=sinkhorn
lradj=type1
train_epochs=10
patience=3
batch_size=32
test_batch_size=1

rerun=0

for mask_factor in ${mask_factor_list[@]}; do
for distance in ${distance_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for joint_forecast in ${joint_forecast_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}
    OUTPUT_DIR="./results_FFTOT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTm1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --decomposition ${decomposition} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --distance ${distance} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 2
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done







# hyper-parameters
dst=ETTm2

# pl_list=(96 192 336 720)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(96 192 336 720)
lbd_list=(1.0 0.8 0.6 0.4 0.2 0.0)
lr_list=(0.001)
auxi_loss_list=(None)


lradj=type1
train_epochs=10
patience=3
batch_size=32
test_batch_size=1

rerun=0


for auxi_loss in ${auxi_loss_list[@]}; do
for use_weights in ${use_weights_list[@]}; do
for reinit in ${reinit_list[@]}; do
for pca_dim in ${pca_dim_list[@]}; do
for rank_ratio in ${rank_ratio_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_FFTOT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm2.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTm2 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --decomposition ${decomposition} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 2
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done





# hyper-parameters
dst=ECL

# pl_list=(96 192 336 720)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(96 192 336 720)
lbd_list=(1.0 0.8 0.6 0.4 0.2 0.0)
lr_list=(0.001)
auxi_loss_list=(None)


lradj=type1
train_epochs=10
patience=3
batch_size=32
test_batch_size=1

rerun=0


for auxi_loss in ${auxi_loss_list[@]}; do
for use_weights in ${use_weights_list[@]}; do
for reinit in ${reinit_list[@]}; do
for pca_dim in ${pca_dim_list[@]}; do
for rank_ratio in ${rank_ratio_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_FFTOT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/electricity/ \
            --data_path electricity.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --decomposition ${decomposition} \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 2
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done





# hyper-parameters
dst=Traffic

# pl_list=(96 192 336 720)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(96 192 336 720)
lbd_list=(1.0 0.8 0.6 0.4 0.2 0.0)
lr_list=(0.001)
auxi_loss_list=(None)



lradj=type1
train_epochs=10
patience=3
batch_size=32
test_batch_size=1

rerun=0


for auxi_loss in ${auxi_loss_list[@]}; do
for use_weights in ${use_weights_list[@]}; do
for reinit in ${reinit_list[@]}; do
for pca_dim in ${pca_dim_list[@]}; do
for rank_ratio in ${rank_ratio_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_FFTOT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/traffic/ \
            --data_path traffic.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --decomposition ${decomposition} \
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 2
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done






# hyper-parameters
dst=Weather

# pl_list=(96 192 336 720)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(96 192 336 720)
lbd_list=(1.0 0.8 0.6 0.4 0.2 0.0)
lr_list=(0.001)
auxi_loss_list=(None)



lradj=type1
train_epochs=10
patience=3
batch_size=32
test_batch_size=1

rerun=0


for auxi_loss in ${auxi_loss_list[@]}; do
for use_weights in ${use_weights_list[@]}; do
for reinit in ${reinit_list[@]}; do
for pca_dim in ${pca_dim_list[@]}; do
for rank_ratio in ${rank_ratio_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_FFTOT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/weather/ \
            --data_path weather.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --decomposition ${decomposition} \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 2
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done





# hyper-parameters
dst=PEMS03

# pl_list=(12 24 36 48)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(12 24 36 48)
lbd_list=(1.0 0.8 0.6 0.4 0.2 0.0)
lr_list=(0.001)
auxi_loss_list=(None)



lradj=type1
train_epochs=10
patience=3
batch_size=32
test_batch_size=1

rerun=0


for auxi_loss in ${auxi_loss_list[@]}; do
for use_weights in ${use_weights_list[@]}; do
for reinit in ${reinit_list[@]}; do
for pca_dim in ${pca_dim_list[@]}; do
for rank_ratio in ${rank_ratio_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_FFTOT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS03.npz \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data PEMS \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --decomposition ${decomposition} \
            --enc_in 358 \
            --dec_in 358 \
            --c_out 358 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 2
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done








# hyper-parameters
dst=PEMS08

# pl_list=(12 24 36 48)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(12 24 36 48)
lbd_list=(1.0 0.8 0.6 0.4 0.2 0.0)
lr_list=(0.001)
auxi_loss_list=(None)



lradj=type1
train_epochs=10
patience=3
batch_size=32
test_batch_size=1

rerun=0


for auxi_loss in ${auxi_loss_list[@]}; do
for use_weights in ${use_weights_list[@]}; do
for reinit in ${reinit_list[@]}; do
for pca_dim in ${pca_dim_list[@]}; do
for rank_ratio in ${rank_ratio_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_FFTOT/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS08.npz \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data PEMS \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --decomposition ${decomposition} \
            --enc_in 170 \
            --dec_in 170 \
            --c_out 170 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 2
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done




wait