#!/bin/bash
MAX_JOBS=4
GPUS=(0 1 2 3)
TOTAL_GPUS=${#GPUS[@]}
export OMP_NUM_THREADS=1

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
EXP_NAME=diff_trans
seed=2023
des='iTransformer'

model_name=iTransformer
auxi_mode=basis
# datasets=(ETTh1 ETTh2 ETTm1 ETTm2 ECL Traffic Weather PEMS03 PEMS08)
# datasets=(ECL_PCA Traffic_PCA PEMS08_PCA)
# datasets=(ECL_RobustPCA ECL_ICA ECL_SVD)
datasets=(ECL_FA)
# datasets=(ETTh1)
test_batch_size=1



# hyper-parameters
dst=ECL_FA
auxi_type=fa

pl_list=(96 192 336 720)
lbd_list=(0.2)
lr_list=(0.001)
rank_ratio_list=(1.0)
pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)

lradj=type1
train_epochs=10
patience=3
batch_size=16

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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}
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
            --data custom_FA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --e_layers 3 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --d_ff 512 \
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
            --speedup_sklearn 1

        sleep 5
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
dst=ECL_RobustPCA
auxi_type=robustpca

pl_list=(96 192 336 720)
lbd_list=(0.2 0.4 0.6 0.8)
lr_list=(0.001)
rank_ratio_list=(1.0)
pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)

lradj=type1
train_epochs=10
patience=3
batch_size=16

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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}
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
            --data custom_RobustPCA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --e_layers 3 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --d_ff 512 \
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
            --speedup_sklearn 1 \
            --thread 2

        sleep 5
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
dst=ECL_SVD
auxi_type=svd

pl_list=(96 192 336 720)
lbd_list=(0.2 0.4 0.6 0.8)
lr_list=(0.001)
rank_ratio_list=(1.0)
pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)

lradj=type1
train_epochs=10
patience=3
batch_size=16

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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}
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
            --data custom_SVD \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --e_layers 3 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --d_ff 512 \
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
            --speedup_sklearn 1 \
            --thread 2

        sleep 5
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
dst=ECL_ICA
auxi_type=ica

pl_list=(96 192 336 720)
lbd_list=(0.2 0.4 0.6 0.8)
lr_list=(0.001)
rank_ratio_list=(1.0)
pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)

lradj=type1
train_epochs=10
patience=3
batch_size=16

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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}
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
            --data custom_ICA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --e_layers 3 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --d_ff 512 \
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
            --speedup_sklearn 1 \
            --thread 2

        sleep 5
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
dst=ECL_RobustICA
auxi_type=robustica

pl_list=(96 192 336 720)
lbd_list=(0.2 0.4 0.6 0.8)
lr_list=(0.001)
rank_ratio_list=(1.0)
pca_dim_list=(T)
reinit_list=(1)
auxi_loss_list=(MAE)
use_weights_list=(0)

lradj=type1
train_epochs=10
patience=3
batch_size=16

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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}
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
            --data custom_RobustICA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --e_layers 3 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --d_ff 512 \
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
            --speedup_sklearn 1 \
            --thread 2

        sleep 5
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