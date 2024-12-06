import os
import subprocess

# 设置路径和模型
ROOT = os.getenv('ROOT')
model = 'tgn'
os.chdir(f"{ROOT}/tgnnexplainer/xgraph/models/ext/tgn")

# 运行次数和数据集
runs = [0]
sim_epochs = 150
sim_datasets=['simulate_v1', 'simulate_v2']

# 进行训练和检查点保存
for run in runs:
    for dataset in sim_datasets:
        print(f"dataset: {dataset}\n")
        
        # 训练模型
        subprocess.run([
            'python', 'train_simulate.py',
            '-d', dataset,
            '--prefix', 'tgn-attn',
            '--n_runs', '1',
            '--n_epoch', str(sim_epochs),
            '--n_layer', '2',
            '--n_degree', '10',
            '--use_memory',
            '--gpu', '0'
        ])
        
        # 保存最后一个epoch的模型
        os.makedirs(f"{ROOT}/tgnnexplainer/xgraph/models/checkpoints", exist_ok=True)
        source_path = f"./checkpoints/tgn-attn-{dataset}-{sim_epochs-1}.pth"
        target_path = f"{ROOT}/tgnnexplainer/xgraph/models/checkpoints/{model}_{dataset}_best.pth"
        subprocess.run(['cp', source_path, target_path])
        print(f"{source_path} {target_path} copied")
