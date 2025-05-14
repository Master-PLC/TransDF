#!/bin/bash
MAX_JOBS=8
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
EXP_NAME=diff_loss

seed=2023
des='iTransformer'
# total_modes=(rfft fourier_koopman dilate_cuda dpp)
total_modes=(rfft dilate_cuda soft_dtw dtw)




module_first=1
train_epochs=10
patience=3
lradj=type1
batch_size=32

rerun=0


lr_list=(0.0005)
pl_list=(96 192 336 720)

model_name=iTransformer
dst="ETTh1"
num_freqs=16
auxi_loss="MAE"
auxi_mode=rfft

for lambda in 0.2 0.4 0.6; do
for lr in ${lr_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${total_modes[@]} " =~ " ${auxi_mode} " ]]; then
        continue
    fi


    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${auxi_mode}
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
    > "${OUTPUT_DIR}/stdout.txt"
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
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
            --batch_size ${batch_size} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --itr 1 \
            --auxi_lambda ${ax} \
            --rec_lambda ${rl} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --train_epochs ${train_epochs} \
            --patience ${patience}

    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done





train_epochs=10
patience=3
lradj=type1
batch_size=32

rerun=0


lr_list=(0.0005 0.001)
pl_list=(96 192 336 720)

model_name=iTransformer
dst="ETTh1"
auxi_loss="None"
auxi_mode=soft_dtw

# for lambda in 0.95; do
for lambda in 0.9995 0.995 0.95; do
for lr in ${lr_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${total_modes[@]} " =~ " ${auxi_mode} " ]]; then
        continue
    fi


    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${auxi_mode}
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
    > "${OUTPUT_DIR}/stdout.txt"
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
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
            --batch_size ${batch_size} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --itr 1 \
            --auxi_lambda ${ax} \
            --rec_lambda ${rl} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --train_epochs ${train_epochs} \
            --patience ${patience}

    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done





train_epochs=10
patience=3
lradj=type1
batch_size=32

rerun=0


lr_list=(0.0005 0.001)
pl_list=(96 192 336 720)

model_name=iTransformer
dst="ETTh1"
auxi_loss="None"
auxi_mode=dtw

# for lambda in 0.95; do
for lambda in 0.9995 0.995 0.95; do
for lr in ${lr_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${total_modes[@]} " =~ " ${auxi_mode} " ]]; then
        continue
    fi


    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${auxi_mode}
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
    > "${OUTPUT_DIR}/stdout.txt"
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
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
            --batch_size ${batch_size} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --itr 1 \
            --auxi_lambda ${ax} \
            --rec_lambda ${rl} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --train_epochs ${train_epochs} \
            --patience ${patience}

    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done





lr_list=(0.0005)
pl_list=(96 192 336 720)

model_name=iTransformer
dst="ETTh1"
num_freqs=16
auxi_mode="fourier_koopman"
auxi_loss="None"

for lambda in 0.2; do
for lr in ${lr_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${total_modes[@]} " =~ " ${auxi_mode} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${auxi_mode}
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
    > "${OUTPUT_DIR}/stdout.txt"
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
            --data ETTh1_Fourier \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
            --batch_size ${batch_size} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --itr 1 \
            --auxi_lambda ${ax} \
            --rec_lambda ${rl} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --num_freqs ${num_freqs}

    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done





lr_list=(0.0005)
pl_list=(96 192 336 720)

model_name=iTransformer
dst="ETTh1"
alpha=0.5
gamma=0.01
auxi_mode="dilate_cuda"
auxi_loss="None"

for lambda in 0.9995; do
for lr in ${lr_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${total_modes[@]} " =~ " ${auxi_mode} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${auxi_mode}
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
    > "${OUTPUT_DIR}/stdout.txt"
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
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
            --batch_size ${batch_size} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --itr 1 \
            --auxi_lambda ${ax} \
            --rec_lambda ${rl} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --alpha ${alpha} \
            --gamma ${gamma}

    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done





lr_list=(0.0005)
pl_list=(96 192 336 720)

model_name=iTransformer
dst="ETTh1"
alpha=0.5
gamma=0.01
auxi_mode='dpp'
auxi_loss="None"

for lambda in 0.9995; do
for lr in ${lr_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${total_modes[@]} " =~ " ${auxi_mode} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${auxi_mode}
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
    > "${OUTPUT_DIR}/stdout.txt"
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
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
            --batch_size ${batch_size} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --itr 1 \
            --auxi_lambda ${ax} \
            --rec_lambda ${rl} \
            --auxi_mode ${auxi_mode} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --alpha ${alpha} \
            --gamma ${gamma}

    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done





wait
