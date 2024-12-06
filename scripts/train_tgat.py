import os
import subprocess

# 设置路径和模型
ROOT = os.getenv('ROOT')
model = 'tgat'
os.chdir(f"{ROOT}/tgnnexplainer/xgraph/models/ext/tgat")

# 创建保存检查点的目录
os.makedirs(f"{ROOT}/tgnnexplainer/xgraph/models/checkpoints", exist_ok=True)

# 数据集和运行次数
real_datasets = ['wikipedia', 'reddit']
runs = [0]
real_epochs = 20

# 进行训练和检查点保存
for run in runs:
    print(f"Iteration no. {run}")
    
    for dataset in real_datasets:
        print(f"dataset: {dataset}")
        
        # 训练模型
        subprocess.run([
            'python', 'learn_edge.py',
            '-d', dataset,
            '--bs', '512',
            '--n_degree', '10',
            '--n_epoch', str(real_epochs),
            '--agg_method', 'attn',
            '--attn_mode', 'prod',
            '--gpu', '1',
            '--n_head', '2',
            '--prefix', dataset
        ])
        
        # 复制检查点
        source_path = f"./checkpoints/{dataset}-attn-prod-{real_epochs-1}.pth"
        target_path = f"{ROOT}/tgnnexplainer/xgraph/models/checkpoints/{model}_{dataset}_best.pth"
        subprocess.run(['cp', source_path, target_path])
        print(f"{source_path} {target_path} copied")

