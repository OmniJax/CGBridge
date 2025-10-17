echo "Loading Anaconda environment..."
source /miniconda3/bin/activate

conda activate  CGBridge 

echo "Python path: $(which python)"
echo "Current conda env: $CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"

cd /path/to/CGBridge/scripts


# --- 执行命令 ---
echo "=== 启动训练 ==="
echo "Starting accelerate launch..."


num_processes=4

accelerate launch --num_processes=$num_processes  \
CGBridge_Stage3_Trainer.py \
--config @stage3_configs/summ-qwencoder1.5b-gt-ACD-2-unixcoder-32q.yaml

echo "=== Stage3 for summarization done ==="



echo "$(date): train Stage3 for summarization"

accelerate launch --num_processes 8 CGBridge_Stage3_Trainer.py --config @stage3_configs/summ-qwencoder1.5b-gt-ACD-2-unixcoder-32q.yaml

echo "$(date): train Stage3 for summarization done"
echo "================================================"

echo "$(date): evaluate Stage3 for summarization"

python evaluate_summaries.py --input /path/to/CGBridge/tasks/summarization/results/ablation/summ-qwencoder1.5b-gt-ACD-2-unixcoder-32q/test_summary.csv 


echo "$(date): evaluate Stage3 for summarization done"
echo "================================================"



echo "$(date): all done!!!"

