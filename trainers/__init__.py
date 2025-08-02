# 添加导入路径处理
import sys
import os

# 将项目根目录添加到sys.path
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# # 注意：只有当其他模块需要使用这些类时，再导入它们
# # from .contrastive_trainer import CodeGraphContrastiveTrainer
# # from .alignment import GraphTextAligner, GraphTextAlignmentTrainer, CodeGraphTextDataset
# from trainers.CodeGraphQformer import CodeGraphQformerTrainer, CodeGraphTextDataset
