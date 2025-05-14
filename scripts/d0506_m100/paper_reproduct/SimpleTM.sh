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
seed=2025
des='SimpleTM'

model_name=SimpleTM
datasets=(ETTh1 ETTh2 ETTm1 ETTm2 Weather)
# datasets=(ETTh1)



# hyper-parameters
dst=ETTh1
pl_list=(96 192 336 720)

lambda=1.0

optim_type=adamw
lr=0.02
lradj=TST
train_epochs=10
patience=3
batch_size=256
test_batch_size=256
use_norm=1
wv=db1
m=3
alpha=0.3
l1_weight=0.0005

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    if [[ $pl -eq 96 ]]; then
        alpha=0.3
        l1_weight=0.0005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.02
    elif [[ $pl -eq 192 ]]; then
        alpha=1.0
        l1_weight=0.00005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.02
    elif [[ $pl -eq 336 ]]; then
        alpha=0.0
        l1_weight=0.0
        e_layers=4
        d_model=64
        d_ff=64
        lr=0.002
    elif [[ $pl -eq 720 ]]; then
        alpha=0.9
        l1_weight=0.0005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.009
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${use_norm}_${wv}_${m}_${alpha}_${l1_weight}
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
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers ${e_layers} \
            --d_model ${d_model} \
            --d_ff ${d_ff} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --optim_type ${optim_type} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --use_norm ${use_norm} \
            --wv ${wv} \
            --m ${m} \
            --alpha ${alpha} \
            --l1_weight ${l1_weight} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --output_attention \
            --kernel_size None

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done





# hyper-parameters
dst=ETTh2
pl_list=(96 192 336 720)

lambda=1.0


lradj=TST
train_epochs=10
patience=3
batch_size=256
test_batch_size=256
use_norm=1
m=1

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi


    if [[ $pl -eq 96 ]]; then
        alpha=0.1
        l1_weight=0.0005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.006
        wv=bior3.1
    elif [[ $pl -eq 192 ]]; then
        alpha=0.1
        l1_weight=0.005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.006
        wv=db1
    elif [[ $pl -eq 336 ]]; then
        alpha=0.9
        l1_weight=0.0
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.003
        wv=db1
    elif [[ $pl -eq 720 ]]; then
        alpha=1.0
        l1_weight=0.00005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.003
        wv=db1
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${use_norm}_${wv}_${m}_${alpha}_${l1_weight}
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
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers ${e_layers} \
            --d_model ${d_model} \
            --d_ff ${d_ff} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --optim_type ${optim_type} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --use_norm ${use_norm} \
            --wv ${wv} \
            --m ${m} \
            --alpha ${alpha} \
            --l1_weight ${l1_weight} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --output_attention \
            --kernel_size None

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done







# hyper-parameters
dst=ETTm1
pl_list=(96 192 336 720)

lambda=1.0

lradj=TST
train_epochs=10
patience=3
batch_size=256
test_batch_size=256
use_norm=1
m=3

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    if [[ $pl -eq 96 ]]; then
        alpha=0.1
        l1_weight=0.005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.02
        wv=db1
        m=3
    elif [[ $pl -eq 192 ]]; then
        alpha=0.1
        l1_weight=0.005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.02
        wv=db1
        m=3
    elif [[ $pl -eq 336 ]]; then
        alpha=0.1
        l1_weight=0.005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.02
        wv=db1
        m=1
    elif [[ $pl -eq 720 ]]; then
        alpha=0.1
        l1_weight=0.005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.02
        wv=db1
        m=3
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${use_norm}_${wv}_${m}_${alpha}_${l1_weight}
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
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers ${e_layers} \
            --d_model ${d_model} \
            --d_ff ${d_ff} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --optim_type ${optim_type} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --use_norm ${use_norm} \
            --wv ${wv} \
            --m ${m} \
            --alpha ${alpha} \
            --l1_weight ${l1_weight} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --output_attention \
            --kernel_size None

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done







# hyper-parameters
dst=ETTm2
pl_list=(96 192 336 720)

lambda=1.0


lradj=TST
train_epochs=10
patience=3
batch_size=128
test_batch_size=128
use_norm=1

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    if [[ $pl -eq 96 ]]; then
        alpha=0.3
        l1_weight=0.0005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.006
        wv=bior3.1
        m=3
    elif [[ $pl -eq 192 ]]; then
        alpha=0.0
        l1_weight=0.005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.006
        wv=bior3.1
        m=1
    elif [[ $pl -eq 336 ]]; then
        alpha=0.6
        l1_weight=0.00005
        e_layers=1
        d_model=64
        d_ff=64
        lr=0.006
        wv=bior3.1
        m=1
    elif [[ $pl -eq 720 ]]; then
        alpha=1.0
        l1_weight=0.0
        e_layers=1
        d_model=96
        d_ff=96
        lr=0.003
        wv=db1
        m=3
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${use_norm}_${wv}_${m}_${alpha}_${l1_weight}
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
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers ${e_layers} \
            --d_model ${d_model} \
            --d_ff ${d_ff} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --optim_type ${optim_type} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --use_norm ${use_norm} \
            --wv ${wv} \
            --m ${m} \
            --alpha ${alpha} \
            --l1_weight ${l1_weight} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --output_attention \
            --kernel_size None

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=ECL
pl_list=(96 192 336 720)

lambda=1.0

lradj=TST
train_epochs=10
patience=3
batch_size=256
test_batch_size=256
use_norm=1


rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    if [[ $pl -eq 96 ]]; then
        alpha=0.0
        l1_weight=0.0
        e_layers=1
        d_model=256
        d_ff=1024
        lr=0.01
        wv=db1
        m=3
    elif [[ $pl -eq 192 ]]; then
        alpha=0.0
        l1_weight=0.0
        e_layers=1
        d_model=256
        d_ff=1024
        lr=0.006
        wv=db1
        m=3
    elif [[ $pl -eq 336 ]]; then
        alpha=0.0
        l1_weight=0.00005
        e_layers=1
        d_model=256
        d_ff=1024
        lr=0.006
        wv=db1
        m=3
    elif [[ $pl -eq 720 ]]; then
        alpha=0.0
        l1_weight=0.00005
        e_layers=1
        d_model=256
        d_ff=1024
        lr=0.006
        wv=db1
        m=3
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${use_norm}_${wv}_${m}_${alpha}_${l1_weight}
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
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers ${e_layers} \
            --d_model ${d_model} \
            --d_ff ${d_ff} \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --des ${des} \
            --learning_rate ${lr} \
            --optim_type ${optim_type} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --use_norm ${use_norm} \
            --wv ${wv} \
            --m ${m} \
            --alpha ${alpha} \
            --l1_weight ${l1_weight} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --output_attention \
            --kernel_size None

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done





# hyper-parameters
dst=Traffic
pl_list=(96 192 336 720)

lambda=1.0

lradj=TST
train_epochs=10
patience=3
batch_size=256
test_batch_size=256
use_norm=1

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    if [[ $pl -eq 96 ]]; then
        alpha=0.0
        l1_weight=0.0
        e_layers=1
        d_model=256
        d_ff=1024
        lr=0.01
        wv=db1
        m=3
    elif [[ $pl -eq 192 ]]; then
        alpha=0.0
        l1_weight=0.0
        e_layers=1
        d_model=256
        d_ff=1024
        lr=0.006
        wv=db1
        m=3
    elif [[ $pl -eq 336 ]]; then
        alpha=0.0
        l1_weight=0.00005
        e_layers=1
        d_model=256
        d_ff=1024
        lr=0.006
        wv=db1
        m=3
    elif [[ $pl -eq 720 ]]; then
        alpha=0.0
        l1_weight=0.00005
        e_layers=1
        d_model=256
        d_ff=1024
        lr=0.006
        wv=db1
        m=3
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${use_norm}_${wv}_${m}_${alpha}_${l1_weight}
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
            --e_layers ${e_layers} \
            --d_model ${d_model} \
            --d_ff ${d_ff} \
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --des ${des} \
            --learning_rate ${lr} \
            --optim_type ${optim_type} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --use_norm ${use_norm} \
            --wv ${wv} \
            --m ${m} \
            --alpha ${alpha} \
            --l1_weight ${l1_weight} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --output_attention \
            --kernel_size None

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=Weather
pl_list=(96 192 336 720)

lambda=1.0

lradj=TST
train_epochs=10
patience=3
batch_size=256
test_batch_size=256
use_norm=1

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    if [[ $pl -eq 96 ]]; then
        alpha=0.3
        l1_weight=0.00005
        e_layers=4
        d_model=32
        d_ff=32
        lr=0.01
        wv=db4
        m=1
    elif [[ $pl -eq 192 ]]; then
        alpha=0.3
        l1_weight=0.0
        e_layers=4
        d_model=32
        d_ff=32
        lr=0.009
        wv=db4
        m=1
    elif [[ $pl -eq 336 ]]; then
        alpha=1.0
        l1_weight=0.00005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.009
        wv=db4
        m=3
    elif [[ $pl -eq 720 ]]; then
        alpha=0.9
        l1_weight=0.005
        e_layers=1
        d_model=32
        d_ff=32
        lr=0.02
        wv=db4
        m=1
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${use_norm}_${wv}_${m}_${alpha}_${l1_weight}
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
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers ${e_layers} \
            --d_model ${d_model} \
            --d_ff ${d_ff} \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --des ${des} \
            --learning_rate ${lr} \
            --optim_type ${optim_type} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --use_norm ${use_norm} \
            --wv ${wv} \
            --m ${m} \
            --alpha ${alpha} \
            --l1_weight ${l1_weight} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --output_attention \
            --kernel_size None

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=PEMS03
pl_list=(12 24 36 48)

lambda=1.0


lr=0.002
lradj=TST
train_epochs=20
patience=10
batch_size=16
test_batch_size=16
use_norm=0

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12) alpha=0.1 l1_weight=0.005 e_layers=1 d_model=256 d_ff=512 lr=0.002 wv=bior3.1 m=3;;
        24) alpha=0.1 l1_weight=0.005 e_layers=1 d_model=256 d_ff=512 lr=0.002 wv=bior3.1 m=3;;
        36) alpha=0.1 l1_weight=0.005 e_layers=1 d_model=256 d_ff=512 lr=0.002 wv=bior3.1 m=3;;
        48) alpha=0.1 l1_weight=0.005 e_layers=1 d_model=256 d_ff=1024 lr=0.002 wv=bior3.1 m=3;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${use_norm}_${wv}_${m}_${alpha}_${l1_weight}
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
            --e_layers ${e_layers} \
            --d_model ${d_model} \
            --d_ff ${d_ff} \
            --enc_in 358 \
            --dec_in 358 \
            --c_out 358 \
            --des ${des} \
            --learning_rate ${lr} \
            --optim_type ${optim_type} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --use_norm ${use_norm} \
            --wv ${wv} \
            --m ${m} \
            --alpha ${alpha} \
            --l1_weight ${l1_weight} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --output_attention \
            --kernel_size None

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done








# hyper-parameters
dst=PEMS08
pl_list=(12 24 36 48)

lambda=1.0


lr=0.00001
lradj=TST
train_epochs=20
patience=10
batch_size=16
test_batch_size=16
use_norm=0

rerun=0


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12) alpha=0.0 l1_weight=0.0 e_layers=1 d_model=256 d_ff=512 lr=0.001 wv=db12 m=3;;
        24) alpha=0.0 l1_weight=0.0 e_layers=1 d_model=256 d_ff=512 lr=0.001 wv=db12 m=3;;
        36) alpha=0.0 l1_weight=0.0 e_layers=1 d_model=256 d_ff=512 lr=0.001 wv=db12 m=3;;
        48) alpha=0.0 l1_weight=0.0 e_layers=1 d_model=256 d_ff=512 lr=0.001 wv=db12 m=3;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${use_norm}_${wv}_${m}_${alpha}_${l1_weight}
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
            --e_layers ${e_layers} \
            --d_model ${d_model} \
            --d_ff ${d_ff} \
            --enc_in 170 \
            --dec_in 170 \
            --c_out 170 \
            --des ${des} \
            --learning_rate ${lr} \
            --optim_type ${optim_type} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --use_norm ${use_norm} \
            --wv ${wv} \
            --m ${m} \
            --alpha ${alpha} \
            --l1_weight ${l1_weight} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --output_attention \
            --kernel_size None

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done




wait