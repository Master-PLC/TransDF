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
EXP_NAME=stf_baselines
seed=2023
des='STF'


models=(Fredformer)





model_name=Fredformer
# hyper-parameters
dst=M4

span_list=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
lr_list=(0.001 0.002 0.0005)
lambda=1.0

lradj=type1
train_epochs=10
patience=3
batch_size=16
test_batch_size=1

rerun=0

for lr in ${lr_list[@]}; do
for span in ${span_list[@]}; do
    if ! [[ " ${models[@]} " =~ " ${model_name} " ]]; then
        continue
    fi

    cf_dim=128
    cf_depth=2
    cf_heads=8
    cf_mlp=96
    cf_head_dim=32
    d_model=24

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${cf_dim}_${cf_depth}_${cf_heads}_${cf_mlp}_${cf_head_dim}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_short_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ -f "${RESULTS}/m4_results/${span}_forecast.csv" ]; then
            echo ">>>>>>> Job: $JOB_NAME with seasonal pattern $span already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME with seasonal pattern $span"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name short_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/m4 \
            --seasonal_patterns ${span} \
            --model_id "m4_${span}" \
            --model ${model_name} \
            --data_id $dst \
            --data m4 \
            --features M \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --d_model $d_model \
            --d_ff 128 \
            --dropout 0.3 \
            --fc_dropout 0.3 \
            --patch_len 4 \
            --stride 4 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --loss 'SMAPE' \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done



model_name=FBM_L
# hyper-parameters
dst=M4

span_list=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
lr_list=(0.005 0.004 0.003 0.002 0.001 0.0005 0.0002 0.0001 0.00005 0.00002)
lambda=1.0

decomposition=0
lradj=type1
train_epochs=10
patience=3
batch_size=16
test_batch_size=1

rerun=0

for lr in ${lr_list[@]}; do
for span in ${span_list[@]}; do
    if ! [[ " ${models[@]} " =~ " ${model_name} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_short_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ -f "${RESULTS}/m4_results/${span}_forecast.csv" ]; then
            echo ">>>>>>> Job: $JOB_NAME with seasonal pattern $span already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME with seasonal pattern $span"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name short_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/m4 \
            --seasonal_patterns ${span} \
            --model_id "m4_${span}" \
            --model ${model_name} \
            --data_id $dst \
            --data m4 \
            --features M \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --decomposition ${decomposition} \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --loss 'SMAPE' \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done








model_name=FreTS
# hyper-parameters
dst=M4

span_list=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
lambda=1.0

lr=0.001
lradj=type1
train_epochs=10
patience=3
batch_size=16
test_batch_size=1

rerun=0


for span in ${span_list[@]}; do

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_short_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ -f "${RESULTS}/m4_results/${span}_forecast.csv" ]; then
            echo ">>>>>>> Job: $JOB_NAME with seasonal pattern $span already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME with seasonal pattern $span"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name short_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/m4 \
            --seasonal_patterns ${span} \
            --model_id "m4_${span}" \
            --model ${model_name} \
            --data_id $dst \
            --data m4 \
            --features M \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --loss 'SMAPE' \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done





model_name=iTransformer
# hyper-parameters
dst=M4

span_list=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
lr_list=(0.001 0.0005 0.0002)
lambda=1.0

lradj=type1
train_epochs=10
patience=3
batch_size=16
test_batch_size=1

rerun=0


for lr in ${lr_list[@]}; do
for span in ${span_list[@]}; do

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_short_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ -f "${RESULTS}/m4_results/${span}_forecast.csv" ]; then
            echo ">>>>>>> Job: $JOB_NAME with seasonal pattern $span already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME with seasonal pattern $span"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name short_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/m4 \
            --seasonal_patterns ${span} \
            --model_id "m4_${span}" \
            --model ${model_name} \
            --data_id $dst \
            --data m4 \
            --features M \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --loss 'SMAPE' \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done







model_name=MICN
# hyper-parameters
dst=M4

span_list=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
lambda=1.0

lr=0.001
lradj=type1
train_epochs=10
patience=3
batch_size=16
test_batch_size=1

rerun=0


for span in ${span_list[@]}; do

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_short_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ -f "${RESULTS}/m4_results/${span}_forecast.csv" ]; then
            echo ">>>>>>> Job: $JOB_NAME with seasonal pattern $span already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME with seasonal pattern $span"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name short_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/m4 \
            --seasonal_patterns ${span} \
            --model_id "m4_${span}" \
            --model ${model_name} \
            --data_id $dst \
            --data m4 \
            --features M \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 32 \
            --d_ff 32 \
            --top_k 5 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --loss 'SMAPE' \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done







model_name=DLinear
# hyper-parameters
dst=M4

span_list=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
lambda=1.0

lr=0.001
lradj=type1
train_epochs=10
patience=3
batch_size=16
test_batch_size=1

rerun=0


for span in ${span_list[@]}; do

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_short_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ -f "${RESULTS}/m4_results/${span}_forecast.csv" ]; then
            echo ">>>>>>> Job: $JOB_NAME with seasonal pattern $span already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME with seasonal pattern $span"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name short_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/m4 \
            --seasonal_patterns ${span} \
            --model_id "m4_${span}" \
            --model ${model_name} \
            --data_id $dst \
            --data m4 \
            --features M \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --loss 'SMAPE' \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done








model_name=FEDformer
# hyper-parameters
dst=M4

span_list=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
lambda=1.0

lr=0.001
lradj=type1
train_epochs=10
patience=3
batch_size=16
test_batch_size=1

rerun=0


for span in ${span_list[@]}; do

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_short_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ -f "${RESULTS}/m4_results/${span}_forecast.csv" ]; then
            echo ">>>>>>> Job: $JOB_NAME with seasonal pattern $span already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME with seasonal pattern $span"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name short_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/m4 \
            --seasonal_patterns ${span} \
            --model_id "m4_${span}" \
            --model ${model_name} \
            --data_id $dst \
            --data m4 \
            --features M \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --loss 'SMAPE' \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done








model_name=Autoformer
# hyper-parameters
dst=M4

span_list=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
lambda=1.0

lr=0.001
lradj=type1
train_epochs=10
patience=3
batch_size=16
test_batch_size=1

rerun=0


for span in ${span_list[@]}; do

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_short_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ -f "${RESULTS}/m4_results/${span}_forecast.csv" ]; then
            echo ">>>>>>> Job: $JOB_NAME with seasonal pattern $span already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME with seasonal pattern $span"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name short_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/m4 \
            --seasonal_patterns ${span} \
            --model_id "m4_${span}" \
            --model ${model_name} \
            --data_id $dst \
            --data m4 \
            --features M \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --loss 'SMAPE' \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done










wait