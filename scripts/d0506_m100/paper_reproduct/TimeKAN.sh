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
EXP_NAME=paper_reproduct
seed=2021
des='TimeKAN'

model_name=TimeKAN
datasets=(ETTh1 ETTh2 ETTm1 ETTm2 Weather)
# datasets=(ETTh1)



# hyper-parameters
dst=ETTh1
pl_list=(96 192 336 720)
# pl_list=(96)

lambda=1.0


lr=0.0001
lradj=TST
train_epochs=100
patience=10
batch_size=128
test_batch_size=128

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  e_layers=2 down_sampling_layers=2 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=0;;
        192)  e_layers=2 down_sampling_layers=1 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=0;;
        336)  e_layers=2 down_sampling_layers=2 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=0;;
        720)  e_layers=2 down_sampling_layers=3 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${e_layers}_${down_sampling_layers}_${down_sampling_window}_${d_model}_${d_ff}_${begin_order}
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
            --data ETTh1 \
            --features M \
            --seq_len 96 \
            --label_len 0 \
            --pred_len ${pl} \
            --e_layers $e_layers \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --d_model $d_model \
            --d_ff $d_ff \
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
            --fix_seed ${seed} \
            --down_sampling_layers $down_sampling_layers \
            --down_sampling_window $down_sampling_window \
            --begin_order $begin_order \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --channel_independence 1 \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done





# hyper-parameters
dst=ETTh2
pl_list=(96 192 336 720)

lambda=1.0


lr=0.0001
lradj=TST
train_epochs=100
patience=10
batch_size=128
test_batch_size=128

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  e_layers=2 down_sampling_layers=3 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=0 batch_size=8;;
        192)  e_layers=1 down_sampling_layers=1 down_sampling_window=2 lr=0.005 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=0 batch_size=8;;
        336)  e_layers=1 down_sampling_layers=3 down_sampling_window=2 lr=0.005 d_model=32 d_ff=32 train_epochs=10 patience=10 begin_order=0 batch_size=8;;
        720)  e_layers=2 down_sampling_layers=3 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=16;;
    esac
    test_batch_size=$batch_size

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${e_layers}_${down_sampling_layers}_${down_sampling_window}_${d_model}_${d_ff}_${begin_order}
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
            --data ETTh2 \
            --features M \
            --seq_len 96 \
            --label_len 0 \
            --pred_len ${pl} \
            --e_layers $e_layers \
            --d_model $d_model \
            --d_ff $d_ff \
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
            --fix_seed ${seed} \
            --down_sampling_layers $down_sampling_layers \
            --down_sampling_window $down_sampling_window \
            --begin_order $begin_order \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --channel_independence 1 \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done







# hyper-parameters
dst=ETTm1
pl_list=(96 192 336 720)

lambda=1.0


lr=0.0001
lradj=TST
train_epochs=100
patience=10
batch_size=128
test_batch_size=128

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  e_layers=2 down_sampling_layers=2 down_sampling_window=2 lr=0.005 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=32;;
        192)  e_layers=2 down_sampling_layers=2 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=32;;
        336)  e_layers=3 down_sampling_layers=2 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=32;;
        720)  e_layers=2 down_sampling_layers=3 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=16;;
    esac
    test_batch_size=$batch_size

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${e_layers}_${down_sampling_layers}_${down_sampling_window}_${d_model}_${d_ff}_${begin_order}
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
            --data ETTm1 \
            --features M \
            --seq_len 96 \
            --label_len 0 \
            --pred_len ${pl} \
            --e_layers $e_layers \
            --d_model $d_model \
            --d_ff $d_ff \
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
            --fix_seed ${seed} \
            --down_sampling_layers $down_sampling_layers \
            --down_sampling_window $down_sampling_window \
            --begin_order $begin_order \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --channel_independence 1 \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done







# hyper-parameters
dst=ETTm2
pl_list=(96 192 336 720)

lambda=1.0


lr=0.0001
lradj=TST
train_epochs=100
patience=10
batch_size=128
test_batch_size=128

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  e_layers=3 down_sampling_layers=2 down_sampling_window=2 lr=0.01 d_model=32 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=128;;
        192)  e_layers=3 down_sampling_layers=2 down_sampling_window=2 lr=0.005 d_model=32 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=128;;
        336)  e_layers=1 down_sampling_layers=1 down_sampling_window=2 lr=0.01 d_model=32 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=128;;
        720)  e_layers=3 down_sampling_layers=3 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=16;;
    esac
    test_batch_size=$batch_size

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${e_layers}_${down_sampling_layers}_${down_sampling_window}_${d_model}_${d_ff}_${begin_order}
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
            --data ETTm2 \
            --features M \
            --seq_len 96 \
            --label_len 0 \
            --pred_len ${pl} \
            --e_layers $e_layers \
            --d_model $d_model \
            --d_ff $d_ff \
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
            --fix_seed ${seed} \
            --down_sampling_layers $down_sampling_layers \
            --down_sampling_window $down_sampling_window \
            --begin_order $begin_order \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --channel_independence 1 \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=ECL
pl_list=(96 192 336 720)

lambda=1.0


lr=0.01
lradj=TST
train_epochs=100
patience=5
batch_size=32

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  e_layers=3 down_sampling_layers=3 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=32;;
        192)  e_layers=3 down_sampling_layers=3 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=32;;
        336)  e_layers=3 down_sampling_layers=3 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=32;;
        720)  e_layers=3 down_sampling_layers=3 down_sampling_window=2 lr=0.01 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=32;;
    esac
    test_batch_size=$batch_size

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${e_layers}_${down_sampling_layers}_${down_sampling_window}_${d_model}_${d_ff}_${begin_order}
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
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 0 \
            --pred_len ${pl} \
            --e_layers $e_layers \
            --d_layers 1 \
            --factor 3 \
            --d_model $d_model \
            --d_ff $d_ff \
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
            --fix_seed ${seed} \
            --down_sampling_layers $down_sampling_layers \
            --down_sampling_window $down_sampling_window \
            --begin_order $begin_order \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --channel_independence 1 \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done





# hyper-parameters
dst=Traffic
pl_list=(96 192 336 720)

lambda=1.0


lr=0.01
lradj=TST
train_epochs=100
patience=5
batch_size=32

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  cf_dim=512 cf_depth=3 cf_heads=8 cf_mlp=512 cf_head_dim=32 d_model=256;;
        192) cf_dim=512 cf_depth=3 cf_heads=8 cf_mlp=512 cf_head_dim=32 d_model=256;;
        336) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=48 d_model=256;;
        720) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=48 d_model=256;;
    esac
    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${cf_dim}_${cf_depth}_${cf_heads}_${cf_mlp}_${cf_head_dim}
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
            --label_len 0 \
            --pred_len ${pl} \
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 48 \
            --stride 48 \
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.2 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 1 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --channel_independence 1 \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=Weather
pl_list=(96 192 336 720)

lambda=1.0


lr=0.001
lradj=TST
train_epochs=100
patience=5
batch_size=128
test_batch_size=128

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi


    case $pl in
        96)  e_layers=3 down_sampling_layers=3 down_sampling_window=2 lr=0.005 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=32;;
        192)  e_layers=3 down_sampling_layers=2 down_sampling_window=2 lr=0.005 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=32;;
        336)  e_layers=3 down_sampling_layers=2 down_sampling_window=2 lr=0.005 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=32;;
        720)  e_layers=3 down_sampling_layers=2 down_sampling_window=2 lr=0.005 d_model=16 d_ff=32 train_epochs=10 patience=10 begin_order=1 batch_size=128;;
    esac
    test_batch_size=$batch_size


    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${e_layers}_${down_sampling_layers}_${down_sampling_window}_${d_model}_${d_ff}_${begin_order}
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
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 0 \
            --pred_len ${pl} \
            --e_layers $e_layers \
            --d_layers 1 \
            --factor 3 \
            --d_model $d_model \
            --d_ff $d_ff \
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
            --fix_seed ${seed} \
            --down_sampling_layers $down_sampling_layers \
            --down_sampling_window $down_sampling_window \
            --begin_order $begin_order \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --channel_independence 1 \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=PEMS03
pl_list=(12 24 36 48)

lambda=1.0


lr=0.00001
lradj=TST
train_epochs=100
patience=10
batch_size=32

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
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
            --label_len 0 \
            --pred_len ${pl} \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --channel_independence 1 \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done








# hyper-parameters
dst=PEMS08
pl_list=(12 24 36 48)

lambda=1.0


lr=0.00001
lradj=TST
train_epochs=100
patience=10
batch_size=32

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
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
            --label_len 0 \
            --pred_len ${pl} \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --channel_independence 1 \
            --rerun $rerun

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done




wait