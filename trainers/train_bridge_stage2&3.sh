
echo "$(date): train Stage2"

cd /path/to/CGBridge/scripts

accelerate launch --num_processes 8 CGBridge_Stage2_Trainer.py --config @stage2_configs/stage2-${graph}-2-unixcoder-32q.yaml

echo "$(date): train Stage2 done"
echo "================================================"

echo "$(date): train Stage3 for translation"


accelerate launch --num_processes 8 CGBridge_Stage3_Trainer.py --config @stage3_configs/trans-qwencoder1.5b-gt-${graph}-2-unixcoder-32q.yaml

echo "$(date): train Stage3 done"
echo "================================================"

echo "$(date): train Stage3 for translation"

cd /path/to/CGBridge/tasks/translation


accelerate launch \
    --num_processes=8 \
    --gpu_ids="0,1,2,3,4,5,6,7" \
    --main_process_port=29500 \
    translate.py \
    --base_path /path/to/CGBridge/tasks/translation/outputs/ablation-graph/trans-qwencoder1.5b-${graph}-2-unixcoder-32q \
    --dataset test \
    --batch_size 32 \
    --output_dir /path/to/CGBridge/tasks/translation/results/ablation-graph/trans-qwencoder1.5b-${graph}-2-unixcoder-32q


echo "$(date): translate done"
echo "================================================"

echo "$(date): evaluate"

python evaluate_translations.py --input /path/to/CGBridge/tasks/translation/results/ablation-graph/trans-qwencoder1.5b-${graph}-2-unixcoder-32q/test_translation.csv



python check_java_compilation.py -t 128 -c generated --index-field index -i  /path/to/CGBridge/tasks/translation/results/ablation-graph/trans-qwencoder1.5b-${graph}-2-unixcoder-32q/test_translation.csv


echo "$(date): evaluate done"
echo "================================================"

echo "$(date): all done!!!"

