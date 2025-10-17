echo "=== 环境设置 ==="
echo "Loading Anaconda environment..."
source /miniconda3/bin/activate

conda activate  CGBridge 

# 检查环境
echo "Python path: $(which python)"
echo "Current conda env: $CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"

cd /path/to/CGBridge/scripts


# --- 执行命令 ---
echo "=== 启动 ==="


echo "$(date): train CGE"

cd /path/to/CGBridge/scripts

graph="AC"  
accelerate launch --num_processes 8 --gpu_ids="0,1,2,3,4,5,6,7" CGE_Trainer_accelerate.py --config /path/to/CGBridge/configs/CGE_configs/gt-${graph}-2.yaml

echo "$(date): train CGE done"
echo "================================================"

echo "$(date): produce CGE"

echo "$(date): for translation"

for dataset in train valid test; do
    python produce.py \
    --model_path /path/to/CGBridge/outputs/checkpoints/gt-${graph}-unixcoder-2/cur_best_model.pt \
    --config_path /path/to/CGBridge/outputs/checkpoints/gt-${graph}-unixcoder-2/config.yaml \
    --code_data_path /path/to/CGBridge/tasks/translation/xlcost_translate_datset/code_datasets/python2java_${dataset}.csv \
    --graph_data_path /path/to/CGBridge/tasks/translation/xlcost_translate_datset/graph_datasets/python2java_${dataset}.pt \
    --output_path /path/to/CGBridge/tasks/translation/xlcost_translate_datset/pair_datasets/trans-${graph}-2-unixcoder/${dataset}_trans-${graph}-2-unixcoder.csv \
    --batch_size 1024 \
    --device cuda:1 \
    --column_name src_code 
done


echo "$(date): for summarization"

for dataset in train valid test; do
    python produce.py \
    --model_path /path/to/CGBridge/outputs/checkpoints/gt-${graph}-unixcoder-2/cur_best_model.pt \
    --config_path /path/to/CGBridge/outputs/checkpoints/gt-${graph}-unixcoder-2/config.yaml \
    --code_data_path /path/to/CGBridge/tasks/summarization/code_datasets/${dataset}_summarization.csv \
    --graph_data_path /path/to/CGBridge/tasks/summarization/graph_datasets/summ-${graph}-2-unixcoder/${dataset}_summarization.pt \
    --output_path /path/to/CGBridge/tasks/summarization/pair_datasets/summ-${graph}-2-unixcoder/${dataset}_summ-${graph}-2-unixcoder.csv \
    --batch_size 1024 \
    --device cuda:0 \
    --column_name code 
done


echo "$(date): produce done"
