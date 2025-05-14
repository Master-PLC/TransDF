#!/bin/bash
MAX_JOBS=16
GPUS=(0 1 2 3 4 5 6 7)
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
EXP_NAME=diff_model
seed=2023
des='FBM_L'

model_name=FBM_L
auxi_mode=basis
auxi_type=pca
# datasets=(ETTh1 ETTh2 ETTm1 ETTm2 ECL Traffic Weather PEMS03 PEMS08)
# datasets=(ETTh1_PCA ETTh2_PCA ETTm1_PCA ETTm2_PCA Weather_PCA)
datasets=(ECL_PCA Weather_PCA)
# datasets=(ETTh1)



# hyper-parameters
dst=ETTh1_PCA

pl_list=(96 192 336 720)
lbd_list=(0.0 0.2 0.4 0.6 0.8)
# lr_list=(0.0002 0.0001 0.00005)
lr_list=(0.0002 0.0001 0.0005)
rank_ratio_list=(0.1 0.2 0.3 0.5 0.6 0.8 0.9 1.0)
bs_list=(64 128)
pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)


decomposition=0
lradj=type1
train_epochs=10
patience=3
test_batch_size=1

rerun=0

for batch_size in ${bs_list[@]}; do
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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

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
            --data ETTh1_PCA \
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
            --rerun $rerun \
            --thread 1 \
            --speedup_sklearn 2

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
done





# hyper-parameters
dst=ETTh2_PCA

pl_list=(96 192 336 720)
lbd_list=(0.0 0.2 0.4 0.6 0.8)
# lr_list=(0.0002 0.0001 0.00005)
lr_list=(0.0002 0.0001 0.0005)
rank_ratio_list=(0.1 0.2 0.3 0.5 0.6 0.8 0.9 1.0)
bs_list=(64 128)
pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)

decomposition=0
lradj=type1
train_epochs=10
patience=3
test_batch_size=1

rerun=0



for batch_size in ${bs_list[@]}; do
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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

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
            --data ETTh2_PCA \
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
            --rerun $rerun \
            --thread 1 \
            --speedup_sklearn 2

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
done






# hyper-parameters
dst=ETTm1_PCA
auxi_mode=basis
auxi_type=pca


pl_list=(96 192 336 720)
lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
lr_list=(0.001 0.0005 0.0001)
rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)


decomposition=0
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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

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
            --data ETTm1_PCA \
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
dst=ETTm2_PCA

pl_list=(96 192 336 720)
lbd_list=(0.0 0.2 0.4 0.6 0.8)
# lr_list=(0.0002 0.0001 0.00005)
lr_list=(0.0002 0.0001 0.0005)
rank_ratio_list=(0.1 0.2 0.3 0.5 0.6 0.8 0.9 1.0)
bs_list=(64 128)
pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)

decomposition=0
lradj=type1
train_epochs=10
patience=3
test_batch_size=1

rerun=0


for batch_size in ${bs_list[@]}; do
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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

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
            --data ETTm2_PCA \
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
            --rerun $rerun \
            --thread 1 \
            --speedup_sklearn 2

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
done





# hyper-parameters
dst=ECL_PCA

# pl_list=(96 192 336 720)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001 0.0005 0.0001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(96 192 336 720)

lbd_list=(0.5 0.6 0.8)
lr_list=(0.001 0.0005 0.0002 0.00001)
rank_ratio_list=(0.6 0.8 1.0)
bs_list=(32 64 128)

# lbd_list=(0.0 0.2 0.4)
# lr_list=(0.001 0.0005 0.0002)
# rank_ratio_list=(0.2 0.6 0.8 1.0)

pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)

decomposition=0
lradj=type1
train_epochs=10
patience=3
test_batch_size=1

rerun=0


for batch_size in ${bs_list[@]}; do
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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

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
            --data custom_PCA \
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
            --rerun $rerun \
            --thread 1 \
            --speedup_sklearn 2

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
done





# hyper-parameters
dst=Traffic

# pl_list=(96 192 336 720)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001 0.0005 0.0001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(96 192 336 720)
lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
lr_list=(0.001 0.0005 0.0001)
rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)


decomposition=0
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
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

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
dst=Weather_PCA

# pl_list=(96 192 336 720)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001 0.0005 0.0001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(96 192 336 720)
# lbd_list=(0.0 0.2 0.4)
# lr_list=(0.001 0.0005 0.0002)
# rank_ratio_list=(0.2 0.6 0.8 1.0)


lbd_list=(0.5 0.6 0.8)
lr_list=(0.001 0.0005 0.0002)
rank_ratio_list=(0.6 0.8 1.0)
bs_list=(32 64 128)

pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)


decomposition=0
lradj=type1
train_epochs=10
patience=3
test_batch_size=1

rerun=0


for batch_size in ${bs_list[@]}; do
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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${decomposition}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

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
            --data custom_PCA \
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
            --rerun $rerun \
            --thread 1 \
            --speedup_sklearn 2

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
done





# hyper-parameters
dst=PEMS03

# pl_list=(12 24 36 48)
# lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
# lr_list=(0.001 0.0005 0.0001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(12 24 36 48)
lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
lr_list=(0.001 0.0005 0.0001)
rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)


decomposition=0
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
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

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
# lr_list=(0.001 0.0005 0.0001)
# rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
# pca_dim_list=(T D all)
# reinit_list=(0 1)
# auxi_loss_list=(MAE MSE)
# use_weights_list=(0 1)


pl_list=(12 24 36 48)
lbd_list=(0.0 0.2 0.4 0.6 0.8 1.0)
lr_list=(0.001 0.0005 0.0001)
rank_ratio_list=(0.2 0.4 0.6 0.8 1.0)
pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)


decomposition=0
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
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

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