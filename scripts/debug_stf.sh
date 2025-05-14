#!/bin/bash
MAX_JOBS=1
GPUS=(6)
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
EXP_NAME=debug
seed=2025
des='DebugSTF'


# hyper-parameters
dst=m4_PCA
model_name=iTransformer
span_list=("Yearly")
lr=0.001

lambda=0.2

auxi_mode=basis
auxi_type=pca
pca_dim=T
use_weights=1
reinit=1
rank_ratio=0.5

auxi_loss=MAE
module_first=1

rerun=1




mkdir -p "${OUTPUT_DIR}/"
for span in ${span_list[@]}; do
    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    # do not add span to the job name
    JOB_NAME=${model_name}_${dst}_${rl}_${ax}_${lr}_${auxi_mode}_${auxi_type}_${pca_dim}_${use_weights}_${reinit}_${rank_ratio}_${auxi_loss}_${module_first}
    OUTPUT_DIR="./results_PCA/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_short_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.txt"
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
            --task_name short_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/m4/ \
            --seasonal_patterns $span \
            --model_id "${dst}_${span}" \
            --model ${model_name} \
            --data $dst \
            --features M \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --des ${des} \
            --d_model 512 \
            --batch_size 16 \
            --learning_rate ${lr} \
            --itr 1 \
            --rec_lambda ${rl} \
            --loss SMAPE \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --use_weights ${use_weights} \
            --reinit ${reinit} \
            --rank_ratio ${rank_ratio} \
            --auxi_loss ${auxi_loss} \
            --module_first ${module_first} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done



wait