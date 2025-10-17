
echo "Loading Anaconda environment..."
source ...

conda activate  CGBridge 

echo "Starting constructing graph..."
cd /path/to/scripts



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pkl_to_graph.py \
--pkl_path /path/to/tasks/summarization/datasets/python_test_summarization.csv \
--output_path /path/to/tasks/summarization/graph_datasets/test_summarization_ACD.pt \
--model_name_or_path unixcoder-base \
--edge_types AST,CFG,DFG \
--batch_size 512 \
--device "cuda" --column_name "code" 